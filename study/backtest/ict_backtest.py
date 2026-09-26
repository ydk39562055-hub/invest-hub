"""
ICT Sweep -> MSS -> FVG x VWAP  백테스트 (5분 봉, 15분 구조)
study/pine/ict_full_strategy.pine 과 같은 규칙을 파이썬으로 옮긴 것.

사용: python ict_backtest.py <5m csv> [--sweep]
CSV: datetime(ISO, tz 포함), open, high, low, close, volume
"""
import sys, math, itertools
import numpy as np
import pandas as pd

NY = "America/New_York"

# ───────────────────────── 파라미터 ─────────────────────────
DEFAULT = dict(
    htf_len=5, len=5, window=12, pb_window=24,
    atr_len=14, atr_mult=1.5, body_pct=0.5, strong_x=1.5,
    use_pd=True, use_asia=True, use_ldn=True, use_ny=True, use_htf=True, use_int=False,
    s_asia=("20:00", "00:00"), s_ldn=("02:00", "05:00"), s_ny=("09:30", "16:00"),
    kz_mode="필수", kz=[("02:00", "05:00"), ("07:00", "10:00"), ("13:30", "16:00")],
    hold_bars=2, max_touch=2, req_overlap=True, tol_atr=0.5, sigma_t=2.0,
    loc_mode="가중", min_score=4,
    stop_mode="스윕 극단값", stop_buf=1.0, tp1_mode="유동성", rr1=1.0, tp2_mode="2σ", rr2=2.0,
    be_mode="TP1 후", be_r=1.0, max_trades=3, eod=("15:55", "16:00"),
)
# stop_mode: 스윕 극단값 | 오더블록 | FVG 반대편
# tp1_mode : 유동성(가장 가까운 풀) | RR
# be_mode  : TP1 후 | 구역 이탈 후(be_r × 리스크만큼 유리하게 간 뒤) | 없음

# ───────────────────────── 유틸 ─────────────────────────
def in_sess(t, start, end):
    m = t.hour * 60 + t.minute
    s = int(start[:2]) * 60 + int(start[3:])
    e = int(end[:2]) * 60 + int(end[3:])
    if e == 0: e = 24 * 60
    return s <= m < e if s < e else (m >= s or m < e)

def pivots(high, low, n):
    """Pine ta.pivothigh/low(n, n) 와 같은 '마지막 확정 피벗' 시계열. 값은 확정 봉(i+n)부터 유효."""
    N = len(high); sh = np.full(N, np.nan); sl = np.full(N, np.nan)
    cur_h = np.nan; cur_l = np.nan
    for i in range(N):
        c = i - n
        if c - n >= 0:
            w = high[c - n:c + n + 1]
            if high[c] == w.max() and (w == high[c]).sum() == 1: cur_h = high[c]
            w = low[c - n:c + n + 1]
            if low[c] == w.min() and (w == low[c]).sum() == 1: cur_l = low[c]
        sh[i] = cur_h; sl[i] = cur_l
    return sh, sl

def rma(x, n):
    out = np.full(len(x), np.nan); a = 1 / n
    s = np.nan
    for i, v in enumerate(x):
        if np.isnan(v): continue
        s = v if np.isnan(s) else s + a * (v - s)
        out[i] = s
    return out

# ───────────────────────── 데이터 준비 ─────────────────────────
def prepare(df, P):
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert(NY)
    o, h, l, c, v = [df[k].values.astype(float) for k in ("open", "high", "low", "close", "volume")]
    N = len(df); ts = df.index
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1)))); tr[0] = h[0] - l[0]
    df["atr"] = rma(tr, P["atr_len"])
    df["sh"], df["sl"] = pivots(h, l, P["len"])

    # HTF(15분) 스윙: 15분 봉 확정 + 1봉 지연 (Pine lookahead_on + [1] 과 동일)
    htf = df[["open", "high", "low", "close"]].resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    hsh, hsl = pivots(htf["high"].values, htf["low"].values, P["htf_len"])
    # 15분 봉 j 의 값은 그 봉이 닫힌 뒤(=j+1 봉 시작) 사용 가능, 여기에 [1] 지연 → j+2 봉 시작부터
    avail = htf.index + pd.Timedelta(minutes=30)
    hh = pd.Series(hsh, index=avail).reindex(ts, method="ffill").values
    hl = pd.Series(hsl, index=avail).reindex(ts, method="ffill").values
    df["hh"], df["hl"] = hh, hl

    # 전일 고/저 (뉴욕 달력일 기준, 직전 거래일)
    day = ts.date
    daily = df.groupby(day).agg(H=("high", "max"), L=("low", "min"))
    prevH = daily["H"].shift(1); prevL = daily["L"].shift(1)
    df["pdh"] = pd.Series(day, index=ts).map(prevH).values
    df["pdl"] = pd.Series(day, index=ts).map(prevL).values

    # 세션 고/저
    for name, key in (("asia", "s_asia"), ("ldn", "s_ldn"), ("ny", "s_ny")):
        st, en = P[key]; H = np.full(N, np.nan); L = np.full(N, np.nan); IN = np.zeros(N, bool)
        ch = np.nan; cl = np.nan; prev_in = False
        for i in range(N):
            ins = in_sess(ts[i], st, en)
            if ins and (not prev_in or np.isnan(ch)): ch, cl = h[i], l[i]
            elif ins: ch, cl = max(ch, h[i]), min(cl, l[i])
            H[i], L[i], IN[i] = ch, cl, ins; prev_in = ins
        df[name + "H"], df[name + "L"], df["in_" + name] = H, L, IN

    df["kz"] = [any(in_sess(t, a, b) for a, b in P["kz"]) for t in ts]
    df["eod"] = [in_sess(t, *P["eod"]) for t in ts]
    # 거래량이 없는 데이터(histdata)는 VWAP 을 단순 평균으로
    if np.nansum(v) == 0: df["volume"] = 1.0
    return df

def attach_m1(df, m1):
    """5분 봉 i 에 속하는 1분 봉 (high, low, close) 배열을 붙인다. 체결 순서 판정용."""
    m1 = m1.copy(); m1.index = pd.to_datetime(m1.index, utc=True).tz_convert(NY)
    key = m1.index.floor("5min")
    grp = {k: g[["high", "low", "close"]].values for k, g in m1.groupby(key)}
    return [grp.get(t) for t in df.index]

# ───────────────────────── 백테스트 ─────────────────────────
def fill_order(d_, bars, stop, tp1, tp2, tp1_done):
    """1분 봉 시퀀스로 손절/TP 도달 순서 판정. 반환: 이벤트 리스트 [(종류, 가격)]"""
    ev = []
    for hh_, ll_, cc_ in bars:
        hit_stop = ll_ <= stop if d_ == 1 else hh_ >= stop
        hit_tp1 = (not tp1_done) and (hh_ >= tp1 if d_ == 1 else ll_ <= tp1)
        hit_tp2 = tp1_done and (hh_ >= tp2 if d_ == 1 else ll_ <= tp2)
        if hit_stop and (hit_tp1 or hit_tp2):
            ev.append(("STOP", stop)); break          # 1분 안에서도 겹치면 손절 우선
        if hit_stop: ev.append(("STOP", stop)); break
        if hit_tp1: ev.append(("TP1", tp1)); tp1_done = True
        if hit_tp2: ev.append(("TP2", tp2)); break
    return ev

def run(df, P, m1bars=None, verbose=False):
    d = df
    o, h, l, c, v = d.open.values, d.high.values, d.low.values, d.close.values, d.volume.values
    atr, sh, sl, hh, hl = d.atr.values, d.sh.values, d.sl.values, d.hh.values, d.hl.values
    pdh, pdl = d.pdh.values, d.pdl.values
    kz, eod, ts = d.kz.values, d.eod.values, d.index
    sess = {k: (d[k + "H"].values, d[k + "L"].values, d["in_" + k].values) for k in ("asia", "ldn", "ny")}
    use = {"asia": P["use_asia"], "ldn": P["use_ldn"], "ny": P["use_ny"]}
    hlc3 = (h + l + c) / 3

    def sw_lo(i, lvl, on): return on and not np.isnan(lvl) and l[i] < lvl and c[i] > lvl and c[i - 1] >= lvl
    def sw_hi(i, lvl, on): return on and not np.isnan(lvl) and h[i] > lvl and c[i] < lvl and c[i - 1] <= lvl

    st = 0; dr = 0; sw_bar = -1; sw_wick = sw_lvl = np.nan; sw_ext = False; sw_nm = ""
    mss_bar = -1; fvg_top = fvg_bot = np.nan; strong = False
    touches = 0; in_zone = False; hold = 0; against = 0
    sPV = sV = sPV2 = 0.0; a_bar = -1
    trades = []; pos = None; trades_today = 0; cur_day = None

    for i in range(max(P["len"] * 2 + 2, 3), len(d)):
        day = ts[i].date()
        if day != cur_day: cur_day = day; trades_today = 0
        vwap = sPV / sV if a_bar >= 0 and sV > 0 else np.nan
        sd = math.sqrt(max(sPV2 / sV - vwap * vwap, 0)) if not np.isnan(vwap) else np.nan

        # ── 포지션 관리 (진입 다음 봉부터, 이 봉의 OHLC 로 체결 판정)
        if pos is not None:
            p = pos; d_ = p["dir"]
            be_now = (P["be_mode"] == "TP1 후" and p["tp1_done"]) or (P["be_mode"] == "구역 이탈 후" and p["be_on"])
            stop = p["entry"] if be_now else p["stop"]
            if P["tp2_mode"] == "2σ" and not np.isnan(p["sd_prev"]):
                tp2 = (max(p["vwap_prev"] + p["sd_prev"] * P["sigma_t"], p["entry"] + p["risk"] * P["rr1"]) if d_ == 1
                       else min(p["vwap_prev"] - p["sd_prev"] * P["sigma_t"], p["entry"] - p["risk"] * P["rr1"]))
            else:
                tp2 = p["entry"] + d_ * p["risk"] * P["rr2"]
            gap_open = (o[i] <= stop) if d_ == 1 else (o[i] >= stop)
            bars = m1bars[i] if m1bars is not None and m1bars[i] is not None else [(h[i], l[i], c[i])]
            closed = False
            if gap_open:
                p["pnl"] += d_ * (o[i] - p["entry"]) * p["qty"]; closed = True; p["exit"] = "STOP" if not p["tp1_done"] else "BE"
            else:
                for kind, px in fill_order(d_, bars, stop, p["tp1"], tp2, p["tp1_done"]):
                    if kind == "STOP":
                        p["pnl"] += d_ * (px - p["entry"]) * p["qty"]; closed = True; p["exit"] = "STOP" if not p["tp1_done"] else "BE"
                    elif kind == "TP1":
                        p["pnl"] += d_ * (px - p["entry"]) * 0.5; p["qty"] = 0.5; p["tp1_done"] = True
                    elif kind == "TP2":
                        p["pnl"] += d_ * (px - p["entry"]) * p["qty"]; closed = True; p["exit"] = "TP2"
            # 본전 이동: 구역 이탈 모드 = 리스크 × be_r 만큼 유리하게 간 뒤
            if not closed and P["be_mode"] == "구역 이탈 후" and not p["be_on"] and d_ * (c[i] - p["entry"]) >= p["risk"] * P["be_r"]:
                p["be_on"] = True
            if not closed and P["eod"] and eod[i]:
                p["pnl"] += d_ * (c[i] - p["entry"]) * p["qty"]; closed = True; p["exit"] = "EOD"
            if closed:
                p["exit_bar"] = i; p["R"] = p["pnl"] / p["risk"]; trades.append(p); pos = None; st = 0
            else:
                p["vwap_prev"], p["sd_prev"] = vwap, sd
        # ── 새 스윕
        if st != 3:
            lo_hits = [(sw_lo(i, pdl[i], P["use_pd"]), "PDL", pdl[i], True)]
            hi_hits = [(sw_hi(i, pdh[i], P["use_pd"]), "PDH", pdh[i], True)]
            for k, nm in (("asia", "Asia"), ("ldn", "London"), ("ny", "NY")):
                H, L, IN = sess[k]
                lo_hits.append((sw_lo(i, L[i], use[k] and not IN[i]), nm + " L", L[i], True))
                hi_hits.append((sw_hi(i, H[i], use[k] and not IN[i]), nm + " H", H[i], True))
            lo_hits.append((sw_lo(i, hl[i], P["use_htf"]), "HTF L", hl[i], True))
            hi_hits.append((sw_hi(i, hh[i], P["use_htf"]), "HTF H", hh[i], True))
            lo_hits.append((sw_lo(i, sl[i], P["use_int"]), "Swing L", sl[i], False))
            hi_hits.append((sw_hi(i, sh[i], P["use_int"]), "Swing H", sh[i], False))
            bull = next((x for x in lo_hits if x[0]), None)
            bear = next((x for x in hi_hits if x[0]), None) if bull is None else None
            hit = bull or bear
            if hit:
                st = 1; dr = 1 if bull else -1; sw_bar = i
                sw_wick = l[i] if bull else h[i]; sw_nm, sw_lvl, sw_ext = hit[1], hit[2], hit[3]
                sPV = sV = sPV2 = 0.0; a_bar = i; mss_bar = -1
                touches = 0; in_zone = False; hold = 0; against = 0; absorbed = False
        if a_bar >= 0:
            sPV += hlc3[i] * v[i]; sV += v[i]; sPV2 += hlc3[i] ** 2 * v[i]
            vwap = sPV / sV if sV > 0 else np.nan
            sd = math.sqrt(max(sPV2 / sV - vwap * vwap, 0)) if not np.isnan(vwap) else np.nan

        # ── MSS
        if st == 1:
            if i - sw_bar > P["window"]: st = 0
            else:
                rng1 = h[i - 1] - l[i - 1]; body1 = abs(c[i - 1] - o[i - 1])
                disp1 = body1 >= atr[i - 1] * P["atr_mult"] and rng1 > 0 and body1 / rng1 >= P["body_pct"]
                strong1 = body1 >= atr[i - 1] * P["atr_mult"] * P["strong_x"]
                if dr == 1 and not np.isnan(sh[i - 1]) and c[i - 1] > sh[i - 1] and c[i - 2] <= sh[i - 1] and disp1 and l[i] > h[i - 2] and i - 1 >= sw_bar:
                    st = 2; mss_bar = i - 1; fvg_bot, fvg_top, strong = h[i - 2], l[i], strong1
                elif dr == -1 and not np.isnan(sl[i - 1]) and c[i - 1] < sl[i - 1] and c[i - 2] >= sl[i - 1] and disp1 and h[i] < l[i - 2] and i - 1 >= sw_bar:
                    st = 2; mss_bar = i - 1; fvg_top, fvg_bot, strong = l[i - 2], h[i], strong1

        # ── 되돌림 대기 → 진입
        if st == 2:
            side = dr * (c[i] - vwap)
            if i - mss_bar > P["pb_window"] or (dr == 1 and c[i] < fvg_bot) or (dr == -1 and c[i] > fvg_top):
                st = 0
            else:
                against = against + 1 if side < 0 else 0
                if against >= P["hold_bars"]: st = 0
            if st == 2:
                tol = atr[i] * P["tol_atr"]
                overlap = (not P["req_overlap"]) or (not np.isnan(vwap) and fvg_bot - tol <= vwap <= fvg_top + tol)
                touch = overlap and ((l[i] <= fvg_top) if dr == 1 else (h[i] >= fvg_bot))
                if touch:
                    if not in_zone: touches += 1
                    in_zone = True
                else:
                    in_zone = False
                if touches > 0: hold = hold + 1 if side > 0 else 0
                mid = (hh[i] + hl[i]) / 2
                loc = P["loc_mode"] != "사용 안 함" and not np.isnan(mid) and ((sw_lvl < mid) if dr == 1 else (sw_lvl > mid))
                kzh = P["kz_mode"] != "사용 안 함" and kz[i]
                score = (2 if sw_ext else 1) + kzh + loc + strong + (touches == 1)
                kz_ok = P["kz_mode"] != "필수" or kz[i]
                loc_ok = P["loc_mode"] != "필수" or loc
                if (1 <= touches <= P["max_touch"] and hold >= P["hold_bars"] and score >= P["min_score"]
                        and kz_ok and loc_ok and trades_today < P["max_trades"] and not eod[i] and pos is None):
                    entry = c[i]
                    if P["stop_mode"] == "스윕 극단값":
                        base = sw_wick
                    elif P["stop_mode"] == "오더블록":
                        # 변위 봉(mss_bar) 직전 반대색 봉들의 극단값. 없으면 변위 봉 자체 극단값
                        j = mss_bar - 1; cand = []
                        while j >= mss_bar - 3 and j > sw_bar - 1:
                            if (dr == 1 and c[j] < o[j]) or (dr == -1 and c[j] > o[j]): cand.append(l[j] if dr == 1 else h[j])
                            elif cand: break
                            j -= 1
                        base = (min(cand) if dr == 1 else max(cand)) if cand else (l[mss_bar] if dr == 1 else h[mss_bar])
                    else:
                        base = fvg_bot if dr == 1 else fvg_top
                    stop = base - dr * P["stop_buf"]
                    risk = abs(entry - stop)
                    if risk <= 0: st = 0; continue
                    # 가장 가까운 유동성 풀 (진입 방향, 최소 0.5R 거리)
                    pools = [hh[i], pdh[i], sess["asia"][0][i], sess["ldn"][0][i], sess["ny"][0][i], sh[i]] if dr == 1 else \
                            [hl[i], pdl[i], sess["asia"][1][i], sess["ldn"][1][i], sess["ny"][1][i], sl[i]]
                    pools = [x for x in pools if not np.isnan(x) and dr * (x - entry) >= risk * 0.5]
                    liq = (min(pools) if dr == 1 else max(pools)) if pools else np.nan
                    tp1 = liq if (P["tp1_mode"] == "유동성" and not np.isnan(liq)) else entry + dr * risk * P["rr1"]
                    pos = dict(dir=dr, entry=entry, stop=stop, risk=risk, tp1=tp1, qty=1.0, pnl=0.0, tp1_done=False, be_on=False,
                               vwap_prev=vwap, sd_prev=sd, bar=i, time=ts[i], sweep=sw_nm, ext=sw_ext, score=score,
                               kz=bool(kz[i]), loc=bool(loc), strong=bool(strong), touch=touches)
                    trades_today += 1; st = 3
    return pd.DataFrame(trades)

# ───────────────────────── 통계 ─────────────────────────
def stats(tr):
    if len(tr) == 0: return dict(n=0)
    R = tr["R"].values; eq = np.cumsum(R); dd = (np.maximum.accumulate(eq) - eq).max()
    wins = R[R > 0]; loss = R[R <= 0]
    pf = wins.sum() / -loss.sum() if len(loss) and loss.sum() < 0 else float("inf")
    return dict(n=len(R), win=round((R > 0).mean() * 100, 1), avgR=round(R.mean(), 2), sumR=round(R.sum(), 1),
                pf=round(pf, 2), maxDD_R=round(dd, 1), pts=round(tr["pnl"].sum(), 1))

def report(tr):
    print("전체:", stats(tr))
    if len(tr) == 0: return
    for col in ("dir", "sweep", "score", "kz", "touch", "exit"):
        g = tr.groupby(col)["R"].agg(["count", "mean", "sum"]).round(2)
        g["win%"] = tr.groupby(col)["R"].apply(lambda x: round((x > 0).mean() * 100, 1))
        print(f"\n[{col}]\n{g.to_string()}")
    print("\n월별 R:", tr.groupby(tr["time"].dt.to_period("M"))["R"].sum().round(1).to_dict())

if __name__ == "__main__":
    path = sys.argv[1]
    raw = pd.read_csv(path, index_col=0)
    P = dict(DEFAULT)
    for a in sys.argv[2:]:
        if "=" in a and not a.startswith("--"):
            k, val = a.split("=", 1); P[k] = type(DEFAULT[k])(val) if not isinstance(DEFAULT[k], (bool, tuple, list)) else (val == "True" if isinstance(DEFAULT[k], bool) else DEFAULT[k])
    df = prepare(raw, P)
    m1 = None
    if "--m1" in sys.argv:
        m1 = attach_m1(df, pd.read_csv(sys.argv[sys.argv.index("--m1") + 1], index_col=0))
        print("1분 봉으로 체결 순서 판정:", sum(x is not None for x in m1), "/", len(m1), "봉 매핑")
    print(f"봉 수 {len(df)}  기간 {df.index[0]} ~ {df.index[-1]}\n")
    tr = run(df, P, m1)
    report(tr)
    tr.to_csv(path.replace(".csv", "_trades.csv"), index=False)

    if "--compare" in sys.argv:
        print("\n\n=== 손절 방식 × 본전 이동 비교 ===")
        rows = []
        for sm in ("스윕 극단값", "오더블록", "FVG 반대편"):
            for bm in ("TP1 후", "구역 이탈 후", "없음"):
                Q = dict(P); Q["stop_mode"] = sm; Q["be_mode"] = bm
                s_ = stats(run(df, Q, m1)); s_.update(stop=sm, be=bm); rows.append(s_)
        print(pd.DataFrame(rows).to_string(index=False))

    if "--sweep" in sys.argv:
        print("\n\n=== 파라미터 스윕 ===")
        grid = dict(min_score=[3, 4, 5], kz_mode=["필수", "가중"], hold_bars=[1, 2], atr_mult=[1.0, 1.5, 2.0], stop_buf=[0.5, 1.0, 2.0])
        rows = []
        for combo in itertools.product(*grid.values()):
            Q = dict(P); Q.update(dict(zip(grid.keys(), combo)))
            if Q["atr_len"] != P["atr_len"]: df = prepare(raw, Q)
            s = stats(run(df, Q)); s.update(dict(zip(grid.keys(), combo))); rows.append(s)
        res = pd.DataFrame(rows).sort_values("sumR", ascending=False)
        print(res.head(15).to_string(index=False))
        print("\n...하위 5\n", res.tail(5).to_string(index=False))
        res.to_csv(path.replace(".csv", "_sweep.csv"), index=False)
