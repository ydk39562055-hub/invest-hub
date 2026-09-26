"""
Strategy B (nq-strategy-b-bot) 를 미래 참조 없이 재구현한 백테스트.
규칙: 15분 강세 FVG 안에 5분 강세 FVG → 5분 봉 저가가 5분 FVG 를 찍고 종가가 상단 위 마감 → 다음 봉 시가 롱.
손절 5분 FVG 하단 - 2pt, 목표 2R, 15:45 강제 청산, 하루 1회. (숏은 대칭, 옵션)
추가 필터: 시간 창, 1H/4H 편향, 직전 스윕, SMT(상관 자산), 세션 VWAP.
사용: python strategy_b_backtest.py <5m csv> [--corr <상관 자산 5m csv>]
"""
import sys, numpy as np, pandas as pd
from ict_backtest import DEFAULT, prepare, attach_smt, in_sess, stats

NY = "America/New_York"
P0 = dict(min_gap=3.0, age5=24, age15=32, buffer=5.0, stop_buf=2.0, rr=2.0, win=("09:30", "12:00"), hard_exit="15:45",
          max_day=1, short=False, mitigate="bottom", bias_mode="없음", need_sweep=False, sweep_bars=12, smt_mode="없음", vwap=False, be_r=0.0, trail=False)

def fvg_list(o, h, l, c, ts, min_gap, allow):
    """(dir, bottom, top, formed_idx) — formed_idx 는 세 번째 봉 인덱스(이 봉이 닫힌 뒤부터 유효)"""
    out = []
    for i in range(2, len(h)):
        if not allow[i]: continue
        if l[i] - h[i - 2] >= min_gap: out.append((1, h[i - 2], l[i], i))
        if l[i - 2] - h[i] >= min_gap: out.append((-1, h[i], l[i - 2], i))
    return out

def run_b(df, P, smt=None):
    o, h, l, c, v = df.open.values, df.high.values, df.low.values, df.close.values, df.volume.values
    ts = df.index; N = len(df); hhmm = np.array([t.hour * 60 + t.minute for t in ts]); day = np.array([t.date() for t in ts])
    b1, b4 = df.bias1h.values, df.bias4h.values
    # 5분 FVG: 06:00~16:00 에 형성된 것만
    allow5 = (hhmm >= 360) & (hhmm < 960)
    f5 = fvg_list(o, h, l, c, ts, P["min_gap"], allow5)
    # 15분 FVG: RTH 15분 봉. 세 번째 15분 봉이 닫힌 뒤(=그 봉 시작 + 15분) 부터 유효
    rth = df[(hhmm >= 570) & (hhmm < 960)]
    x15 = rth[["open", "high", "low", "close"]].resample("15min", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    o15, h15, l15, c15 = [x15[k].values for k in ("open", "high", "low", "close")]
    f15 = []
    for j in range(2, len(x15)):
        avail = x15.index[j] + pd.Timedelta(minutes=15)
        k = ts.searchsorted(avail)                       # 이 5분 봉 인덱스부터 사용 가능
        if l15[j] - h15[j - 2] >= P["min_gap"]: f15.append((1, h15[j - 2], l15[j], k))
        if l15[j - 2] - h15[j] >= P["min_gap"]: f15.append((-1, h15[j], l15[j - 2], k))
    # 세션 VWAP
    sess_vwap = np.full(N, np.nan); pv = vv = 0.0; cur = None
    for i in range(N):
        if day[i] != cur: cur = day[i]; pv = vv = 0.0
        if hhmm[i] >= 570:
            pv += (h[i] + l[i] + c[i]) / 3 * max(v[i], 1); vv += max(v[i], 1); sess_vwap[i] = pv / vv
    # 직전 스윕 (기준 자산 레벨)
    lv = {k: df[k].values for k in ("pdl", "pdh", "asiaL", "asiaH", "ldnL", "ldnH", "nyL", "nyH", "hl", "hh")}
    def swept_low(i):
        for k in ("pdl", "asiaL", "ldnL", "hl"):
            x = lv[k][i]
            if not np.isnan(x) and l[i] < x and c[i] > x and c[i - 1] >= x: return True
        return False
    def swept_high(i):
        for k in ("pdh", "asiaH", "ldnH", "hh"):
            x = lv[k][i]
            if not np.isnan(x) and h[i] > x and c[i] < x and c[i - 1] <= x: return True
        return False
    if smt is not None: S = {k: smt[k].values for k in smt.columns}
    def smt_ok(i, d):
        if smt is None: return True
        if d == 1:
            for k in ("pdl", "asiaL", "ldnL", "hl"):
                x = lv[k][i]
                if not np.isnan(x) and l[i] < x and c[i] > x:      # 기준 자산이 이 레벨을 스윕한 봉
                    xc = S[k][i]; return (not np.isnan(xc)) and S["low"][i] > xc
            return False
        else:
            for k in ("pdh", "asiaH", "ldnH", "hh"):
                x = lv[k][i]
                if not np.isnan(x) and h[i] > x and c[i] < x:
                    xc = S[k][i]; return (not np.isnan(xc)) and S["high"][i] < xc
            return False

    ws, we = [int(x[:2]) * 60 + int(x[3:]) for x in P["win"]]; hx = int(P["hard_exit"][:2]) * 60 + int(P["hard_exit"][3:])
    p5 = p15 = 0; act5 = []; act15 = []
    trades = []; pos = None; dcount = {}; last_sweep = {1: -10**9, -1: -10**9}
    for i in range(3, N):
        # 포지션 관리 (5분 봉, 손절 우선)
        if pos is not None:
            d = pos["dir"]
            if P["be_r"] > 0 and not pos.get("be") and d * (c[i - 1] - pos["entry"]) >= pos["risk"] * P["be_r"]:
                pos["stop"] = pos["entry"]; pos["be"] = True
            if (o[i] <= pos["stop"]) if d == 1 else (o[i] >= pos["stop"]):
                px, ex = o[i], "STOP"
            elif (l[i] <= pos["stop"]) if d == 1 else (h[i] >= pos["stop"]):
                px, ex = pos["stop"], "STOP"
            elif (h[i] >= pos["tp"]) if d == 1 else (l[i] <= pos["tp"]):
                px, ex = pos["tp"], "TP"
            elif hhmm[i] >= hx or day[i] != pos["day"]:
                px, ex = c[i], "EOD"
            else:
                px = None
            if px is not None:
                if ex == "STOP" and pos.get("be"): ex = "BE"
                r = d * (px - pos["entry"]) / pos["risk"]; pos.update(exit=ex, R=r, pnl=d * (px - pos["entry"])); trades.append(pos); pos = None
        # 활성 FVG 갱신
        while p5 < len(f5) and f5[p5][3] < i: act5.append(f5[p5]); p5 += 1        # 형성 봉 다음부터
        while p15 < len(f15) and f15[p15][3] <= i: act15.append(f15[p15]); p15 += 1
        def alive(lst, age):
            keep = []
            for d, bt, tp, k in lst:
                if i - k > age: continue
                if P["mitigate"] == "bottom":
                    if (d == 1 and c[i] < bt) or (d == -1 and c[i] > tp): continue
                else:
                    m = (bt + tp) / 2
                    if (d == 1 and c[i] < m) or (d == -1 and c[i] > m): continue
                keep.append((d, bt, tp, k))
            return keep
        act5 = alive(act5, P["age5"]); act15 = alive(act15, P["age15"] * 3)
        if swept_low(i): last_sweep[1] = i
        if swept_high(i): last_sweep[-1] = i
        if pos is not None or not (ws <= hhmm[i] < we) or dcount.get(day[i], 0) >= P["max_day"] or i + 1 >= N: continue
        for d in ((1, -1) if P["short"] else (1,)):
            # 필터
            if P["bias_mode"] == "1H" and b1[i] != d: continue
            if P["bias_mode"] == "1H+4H" and not (b1[i] == d and b4[i] == d): continue
            if P["need_sweep"] and i - last_sweep[d] > P["sweep_bars"]: continue
            if P["vwap"] and not np.isnan(sess_vwap[i]) and d * (c[i] - sess_vwap[i]) <= 0: continue
            hit = None
            for d5, bt5, tp5, k5 in act5:
                if d5 != d or k5 >= i: continue
                inside = any(d15 == d and bt5 >= bt15 - (0 if d == 1 else P["buffer"]) and tp5 <= tp15 + (P["buffer"] if d == 1 else 0) for d15, bt15, tp15, k15 in act15)
                if not inside: continue
                if d == 1 and bt5 <= l[i] <= tp5 and c[i] > tp5: hit = (bt5, tp5); break
                if d == -1 and bt5 <= h[i] <= tp5 and c[i] < bt5: hit = (bt5, tp5); break
            if hit is None: continue
            if P["smt_mode"] == "필수":
                # 최근 sweep_bars 안의 스윕 봉에서 SMT 였는지
                ok = any(smt_ok(j, d) for j in range(max(3, i - P["sweep_bars"]), i + 1))
                if not ok: continue
            entry = o[i + 1]
            stop = hit[0] - P["stop_buf"] if d == 1 else hit[1] + P["stop_buf"]
            risk = abs(entry - stop)
            if risk <= 0: continue
            pos = dict(dir=d, entry=entry, stop=stop, risk=risk, tp=entry + d * risk * P["rr"], day=day[i + 1], time=ts[i + 1], bias=int(b1[i]))
            dcount[day[i]] = dcount.get(day[i], 0) + 1
            break
    return pd.DataFrame(trades)

def report(name, tr):
    s = stats(tr); print(name, {k: v for k, v in s.items()}, flush=True)
    if len(tr):
        print("   청산별:", tr.groupby("exit")["R"].agg(["count", "mean"]).round(2).to_dict("index"))
        print("   연도별:", tr.groupby(tr["time"].dt.year)["R"].sum().round(1).to_dict(), flush=True)

if __name__ == "__main__":
    raw = pd.read_csv(sys.argv[1], index_col=0)
    P = dict(DEFAULT); df = prepare(raw, P)
    smt = None
    if "--corr" in sys.argv:
        smt = attach_smt(df, pd.read_csv(sys.argv[sys.argv.index("--corr") + 1], index_col=0), P)
    print(f"봉 수 {len(df)} 기간 {df.index[0]} ~ {df.index[-1]}", flush=True)
    configs = {
        "원본 규칙 (롱 전용, 09:30-12:00)": {},
        "원본 + 숏 허용": dict(short=True),
        "10:00-12:00 창": dict(win=("10:00", "12:00")),
        "1H 편향": dict(bias_mode="1H"),
        "1H+4H 편향": dict(bias_mode="1H+4H"),
        "1H+4H 편향 + 숏": dict(bias_mode="1H+4H", short=True),
        "직전 스윕 필수(12봉)": dict(need_sweep=True),
        "1H+4H + 스윕": dict(bias_mode="1H+4H", need_sweep=True),
        "세션 VWAP 위": dict(vwap=True),
        "1H+4H + VWAP + 10-12시": dict(bias_mode="1H+4H", vwap=True, win=("10:00", "12:00")),
        "하루 3회 허용": dict(max_day=3),
    }
    if smt is not None:
        configs["SMT 필수"] = dict(smt_mode="필수")
        configs["1H+4H + SMT 필수"] = dict(bias_mode="1H+4H", smt_mode="필수")
        configs["1H+4H + SMT + 숏"] = dict(bias_mode="1H+4H", smt_mode="필수", short=True)
    if "--rr" in sys.argv:
        base = dict(P0); base.update(bias_mode="1H+4H", vwap=True, win=("10:00", "12:00"))
        configs = {}
        for rr in (1.0, 1.5, 2.0, 3.0, 4.0):
            configs[f"RR 1:{rr}"] = dict(rr=rr)
            configs[f"RR 1:{rr} + 본전@+1R"] = dict(rr=rr, be_r=1.0)
        for sb in (0.0, 5.0, 10.0):
            configs[f"RR 1:2, 손절여유 {sb}pt"] = dict(rr=2.0, stop_buf=sb)
        configs["RR 1:3, 본전@+1.5R"] = dict(rr=3.0, be_r=1.5)
        configs["RR 1:2, 15:45 대신 종가 보유 안 함(12:00 청산)"] = dict(rr=2.0, hard_exit="12:00")
        configs["RR 1:2, 숏 포함"] = dict(rr=2.0, short=True)
        configs["RR 1:3, 숏 포함, 본전@+1R"] = dict(rr=3.0, short=True, be_r=1.0)
        for name, cfg in configs.items():
            Q = dict(base); Q.update(cfg); report(name, run_b(df, Q, smt))
        print("DONE"); sys.exit()
    for name, cfg in configs.items():
        Q = dict(P0); Q.update(cfg); report(name, run_b(df, Q, smt))
    print("DONE")
