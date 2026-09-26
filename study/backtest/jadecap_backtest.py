"""
JadeCap 'Daily Sweep / SFP' 전략 재구현 (미래 참조 없음)
 1) 편향: 1H 구조(bias1h) 또는 전날 방향(prev-day close-open)
 2) 레벨: 전날~현재까지 1H 스윙 저점(롱) / 고점(숏), 3봉 프랙탈, 1H 봉 마감 후 확정
 3) 09:30 이후 1H 봉이 레벨을 꼬리로 넘고 종가는 되돌린 봉 = SFP (1H 봉 마감 시점에 확정)
 4) SFP 뒤 형성된 5분 FVG 되돌림 진입(다음 봉 시가), 손절 FVG 가운데 봉 극단값 - 2pt, 목표 2R (트레일 옵션)
"""
import sys, numpy as np, pandas as pd
from ict_backtest import DEFAULT, prepare, stats
from strategy_b_backtest import report

P0 = dict(bias="1H", win=("09:30", "12:00"), hard_exit="15:45", rr=2.0, stop_buf=2.0, min_gap=1.0, fvg_wait=24, max_day=1,
          short=True, trail_step=0.0, be_r=0.0, sfp_lookback_days=2, sth_len=1, skip_first_hour=False, tp_mode="RR",
          vwap_filter=False, first_sfp_only=False, min_sweep_atr=0.0, stop_mode="FVG", entry_mode="FVG", partial=False)
# bias: 1H | prev | both / stop_mode: FVG(가운데 봉) | SFP(스윕 꼬리) / entry_mode: FVG(되돌림) | 즉시(SFP 확정 직후 시가) / partial: 1R 에서 50%, 나머지 목표   # tp_mode: RR | 유동성(반대편 세션·전일·1H 스윙 극단값 중 가장 가까운 것, 최소 1R)

def run_j(df, P):
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    ts = df.index; N = len(df); hhmm = np.array([t.hour * 60 + t.minute for t in ts]); day = np.array([t.date() for t in ts])
    b1 = df.bias1h.values
    v = df.volume.values
    sess_vwap = np.full(N, np.nan); pv = vv = 0.0; cur = None
    for i in range(N):
        if day[i] != cur: cur = day[i]; pv = vv = 0.0
        if hhmm[i] >= 570:
            pv += (h[i] + l[i] + c[i]) / 3 * max(v[i], 1); vv += max(v[i], 1); sess_vwap[i] = pv / vv
    pools_hi = [df[k].values for k in ('pdh', 'asiaH', 'ldnH', 'hh')]; pools_lo = [df[k].values for k in ('pdl', 'asiaL', 'ldnL', 'hl')]
    def liq_tp(i, d, entry, risk):
        cand = [x[i] for x in (pools_hi if d == 1 else pools_lo) if not np.isnan(x[i]) and d * (x[i] - entry) >= risk]
        return (min(cand) if d == 1 else max(cand)) if cand else entry + d * risk * P['rr']
    # 전날 방향
    dd = pd.Series(c, index=ts).groupby(day).agg(["first", "last"]); pdir = np.sign(dd["last"] - dd["first"]).shift(1)
    pbias = pd.Series(day, index=ts).map(pdir).fillna(0).values
    # 1H 봉
    x = df[["open", "high", "low", "close"]].resample("60min", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    oh, hh, lh, ch = [x[k].values for k in ("open", "high", "low", "close")]; xt = x.index
    tr1 = np.maximum(hh - lh, np.maximum(abs(hh - np.roll(ch, 1)), abs(lh - np.roll(ch, 1)))); atr1 = pd.Series(tr1).rolling(14).mean().values
    n = P["sth_len"]
    # 1H 스윙 저/고점 목록 (확정 시각 = 피벗 봉 + n 봉 마감)
    swl = [(xt[j + n] + pd.Timedelta(hours=1), lh[j]) for j in range(n, len(x) - n) if lh[j] == lh[j - n:j + n + 1].min()]
    swh = [(xt[j + n] + pd.Timedelta(hours=1), hh[j]) for j in range(n, len(x) - n) if hh[j] == hh[j - n:j + n + 1].max()]
    # 1H SFP 이벤트: (확정 시각, 방향, 레벨, 봉 극단값)
    events = []
    for j in range(len(x)):
        t0 = xt[j]; t1 = t0 + pd.Timedelta(hours=1); lb = t0 - pd.Timedelta(days=P["sfp_lookback_days"])
        loc = t0.hour * 60 + t0.minute
        if loc < 540 or loc >= 720: continue                     # 09:00~11:00 에 시작한 1H 봉 (10:00 마감 SFP 포함)
        lows = [lv for (tc, lv) in swl if lb <= tc <= t0 and lv > lh[j] * 0]
        for lv in lows:
            if lh[j] < lv and ch[j] > lv and oh[j] >= lv and (lv - lh[j]) >= P["min_sweep_atr"] * (atr1[j] if not np.isnan(atr1[j]) else 0): events.append((t1, 1, lv, lh[j])); break
        highs = [lv for (tc, lv) in swh if lb <= tc <= t0]
        for lv in highs:
            if hh[j] > lv and ch[j] < lv and oh[j] <= lv and (hh[j] - lv) >= P["min_sweep_atr"] * (atr1[j] if not np.isnan(atr1[j]) else 0): events.append((t1, -1, lv, hh[j])); break
    events.sort()
    ws, we = [int(v[:2]) * 60 + int(v[3:]) for v in P["win"]]; hx = int(P["hard_exit"][:2]) * 60 + int(P["hard_exit"][3:])
    trades = []; pos = None; dcount = {}; sfp_seen = {}; ei = 0; active = None; F = {"sfp_bias_ok": {1: 0, -1: 0}, "entry": {1: 0, -1: 0}}   # active: (dir, sfp_idx, sfp_wick, deadline)
    for i in range(3, N):
        if pos is not None:
            d = pos["dir"]
            if P["be_r"] > 0 and not pos.get("be") and d * (c[i - 1] - pos["entry"]) >= pos["risk"] * P["be_r"]: pos["stop"] = pos["entry"]; pos["be"] = True
            if P["trail_step"] > 0:
                reached = d * ((h[i - 1] if d == 1 else l[i - 1]) - pos["entry"]) / pos["risk"]; k = int(reached // P["trail_step"])
                if k >= 1:
                    ns = pos["entry"] + d * pos["risk"] * max(0.0, (k - 1) * P["trail_step"])
                    if d * (ns - pos["stop"]) > 0: pos["stop"] = ns; pos["be"] = True
            if P["partial"] and not pos.get("p1") and ((h[i] >= pos["entry"] + pos["risk"]) if d == 1 else (l[i] <= pos["entry"] - pos["risk"])) and not ((l[i] <= pos["stop"]) if d == 1 else (h[i] >= pos["stop"])):
                pos["p1"] = True; pos["bank"] = 0.5 * 1.0; pos["qty"] = 0.5; pos["stop"] = pos["entry"]; pos["be"] = True
            if (o[i] <= pos["stop"]) if d == 1 else (o[i] >= pos["stop"]): px, ex = o[i], "STOP"
            elif (l[i] <= pos["stop"]) if d == 1 else (h[i] >= pos["stop"]): px, ex = pos["stop"], "STOP"
            elif (h[i] >= pos["tp"]) if d == 1 else (l[i] <= pos["tp"]): px, ex = pos["tp"], "TP"
            elif hhmm[i] >= hx or day[i] != pos["day"]: px, ex = c[i], "EOD"
            else: px = None
            if px is not None:
                if ex == "STOP" and pos.get("be"): ex = "BE"
                q = pos.get("qty", 1.0); r_ = pos.get("bank", 0.0) + q * d * (px - pos["entry"]) / pos["risk"]
                pos.update(exit=ex, R=r_, pnl=r_ * pos["risk"]); trades.append(pos); pos = None
        # 새 SFP 확정 (1H 봉 마감 시각 == 이 5분 봉 시작 시각)
        while ei < len(events) and events[ei][0] <= ts[i]:
            t1, d, lv, wick = events[ei]; ei += 1
            if t1 != ts[i]: continue
            bias = b1[i] if P["bias"] == "1H" else (pbias[i] if P["bias"] == "prev" else (b1[i] if b1[i] == pbias[i] else 0))
            if bias != d: continue
            if P["first_sfp_only"] and dcount.get(day[i], 0) + sfp_seen.get(day[i], 0) >= 1: continue
            sfp_seen[day[i]] = sfp_seen.get(day[i], 0) + 1
            if d == -1 and not P["short"]: continue
            if P["skip_first_hour"] and hhmm[i] <= 630: continue
            active = (d, i, wick, i + P["fvg_wait"]); F["sfp_bias_ok"][d] += 1
        if pos is not None or active is None or i + 1 >= N: continue
        d, si, wick, dl = active
        if i > dl or dcount.get(day[i], 0) >= P["max_day"] or not (ws <= hhmm[i] < we): active = None; continue
        if P["vwap_filter"] and not np.isnan(sess_vwap[i]) and d * (c[i] - sess_vwap[i]) <= 0: continue
        if P["entry_mode"] == "즉시":
            entry = o[i + 1]; stop = wick - d * P["stop_buf"]; risk = d * (entry - stop)
            if risk > 0:
                tp = liq_tp(i, d, entry, risk) if P["tp_mode"] == "유동성" else entry + d * risk * P["rr"]
                pos = dict(dir=d, entry=entry, stop=stop, risk=risk, tp=tp, day=day[i + 1], time=ts[i + 1]); dcount[day[i]] = dcount.get(day[i], 0) + 1
            active = None; continue
        # SFP 이후 형성된 5분 FVG 중 가장 최근 것 (형성 봉 k, 가운데 봉 k-1), 되돌림 판정
        for k in range(i - 1, si, -1):
            if d == 1 and l[k] - h[k - 2] >= P["min_gap"]:
                bt, tp, mid_lo = h[k - 2], l[k], l[k - 1]
                if bt <= l[i] <= tp and c[i] > bt:
                    entry = o[i + 1]; stop = (wick if P["stop_mode"] == "SFP" else mid_lo) - P["stop_buf"]; risk = entry - stop
                    if risk > 0: pos = dict(dir=1, entry=entry, stop=stop, risk=risk, tp=(liq_tp(i, 1, entry, risk) if P['tp_mode'] == '유동성' else entry + risk * P["rr"]), day=day[i + 1], time=ts[i + 1]); dcount[day[i]] = dcount.get(day[i], 0) + 1; active = None; F["entry"][1] += 1
                break
            if d == -1 and l[k - 2] - h[k] >= P["min_gap"]:
                bt, tp, mid_hi = h[k], l[k - 2], h[k - 1]
                if bt <= h[i] <= tp and c[i] < tp:
                    entry = o[i + 1]; stop = (wick if P["stop_mode"] == "SFP" else mid_hi) + P["stop_buf"]; risk = stop - entry
                    if risk > 0: pos = dict(dir=-1, entry=entry, stop=stop, risk=risk, tp=(liq_tp(i, -1, entry, risk) if P['tp_mode'] == '유동성' else entry - risk * P["rr"]), day=day[i + 1], time=ts[i + 1]); dcount[day[i]] = dcount.get(day[i], 0) + 1; active = None
                break
    out = pd.DataFrame(trades); out.attrs["F"] = F
    return out

if __name__ == "__main__":
    raw = pd.read_csv(sys.argv[1], index_col=0); df = prepare(raw, dict(DEFAULT))
    print(f"봉 수 {len(df)} 기간 {df.index[0]} ~ {df.index[-1]}", flush=True)
    configs = {
        "JadeCap 원형: 1H 편향, 1:2, 롱+숏": {},
        "롱 전용": dict(short=False),
        "전날 방향 편향": dict(bias="prev"),
        "1:2 + 본전@+1R": dict(be_r=1.0),
        "1:3": dict(rr=3.0),
        "트레일 1R, 목표 없음": dict(rr=99.0, trail_step=1.0),
        "트레일 0.5R + 목표 1:3": dict(rr=3.0, trail_step=0.5),
        "10:30 이전 SFP 제외(첫 시간 트랩 회피)": dict(skip_first_hour=True),
        "스윙 5봉 프랙탈": dict(sth_len=2),
        "하루 2회": dict(max_day=2),
        "12시 이후 진입 창 15:00 까지": dict(win=("09:30", "15:00")),
        "EA 기본: 목표 반대편 유동성": dict(tp_mode="유동성"),
        "EA 기본 + 본전@+1R": dict(tp_mode="유동성", be_r=1.0),
        "손절 여유 5pt, 1:2": dict(stop_buf=5.0),
        "손절 여유 5pt, 1:3 + 본전@+1R": dict(stop_buf=5.0, rr=3.0, be_r=1.0),
        "1:3 + 본전@+1R": dict(rr=3.0, be_r=1.0),
        "트레일 1R + 목표 1:3": dict(rr=3.0, trail_step=1.0),
    }
    if "--improve" in sys.argv:
        base = dict(bias="prev")
        configs = {
            "기준: 전날 편향, 1:2": {},
            "1) 전날+1H 편향 일치": dict(bias="both"),
            "2) 세션 VWAP 필터": dict(vwap_filter=True),
            "3) 하루 첫 SFP 만": dict(first_sfp_only=True),
            "4) 스윕 깊이 ≥ 0.2 ATR(1H)": dict(min_sweep_atr=0.2),
            "5) 손절 SFP 꼬리 아래": dict(stop_mode="SFP"),
            "6) 즉시 진입(SFP 확정 직후)": dict(entry_mode="즉시"),
            "6b) 즉시 진입 + 1:3": dict(entry_mode="즉시", rr=3.0),
            "7) 부분 청산 1R 50% + 2R": dict(partial=True),
            "8) 1+2+3 조합": dict(bias="both", vwap_filter=True, first_sfp_only=True),
            "9) 2+3 조합": dict(vwap_filter=True, first_sfp_only=True),
            "10) 2+3+7 조합": dict(vwap_filter=True, first_sfp_only=True, partial=True),
            "11) 2+3, 1:3": dict(vwap_filter=True, first_sfp_only=True, rr=3.0),
            "12) 2+3+4": dict(vwap_filter=True, first_sfp_only=True, min_sweep_atr=0.2),
        }
        for name, cfg in configs.items():
            Q = dict(P0); Q.update(base); Q.update(cfg); tr = run_j(df, Q); report(name, tr)
        print("DONE"); sys.exit()
    for name, cfg in configs.items():
        Q = dict(P0); Q.update(cfg); tr = run_j(df, Q); print('   깔때기:', tr.attrs['F']); report(name, tr)
    print("DONE")
