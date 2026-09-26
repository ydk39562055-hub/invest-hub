"""
JadeCap 롱 전용 거래 분포로 (1) 100만 원 복리, (2) 프롭펌(Apex 50K 근사) 평가·실계좌 몬테카를로.
사용: python jadecap_money_sim.py <NQ 5m csv>
"""
import sys, numpy as np, pandas as pd
import jadecap_backtest as J
from ict_backtest import DEFAULT, prepare

raw = pd.read_csv(sys.argv[1], index_col=0); df = prepare(raw, dict(DEFAULT))
P = dict(J.P0); P.update(bias="1H", short=False)
tr = J.run_j(df, P).sort_values("time"); R = tr["R"].values; risk = tr["risk"].values
FX = 1400.0; start = 1_000_000 / FX
print(f"거래 {len(R)} 합계R {R.sum():.1f} 평균R {R.mean():.3f} 승률 {(R>0).mean()*100:.1f}% 손절폭 중앙값 {np.median(risk):.1f}pt")

print("\n[100만 원 복리: 자본 비율 리스크]")
for f in (0.01, 0.02, 0.03, 0.05, 0.10):
    eq = start; peak = eq; mdd = 0
    for r in R: eq *= 1 + f * r; peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
    print(f" 리스크 {f*100:>3.0f}% → {eq*FX/1e4:,.0f}만원 ({eq/start:.2f}배) 최대낙폭 {mdd*100:.0f}%")
print("[100만 원, MNQ 정수 계약(최소 1), $2/pt, 왕복 수수료 $1.5]")
for f in (0.02, 0.05):
    eq = start; peak = eq; mdd = 0
    for r, rk in zip(R, risk):
        n = max(1, int(eq * f / (rk * 2))); eq += r * rk * 2 * n - 1.5 * n; peak = max(peak, eq); mdd = max(mdd, 1 - eq / peak)
    print(f" 리스크 {f*100:.0f}% 목표 → {eq*FX/1e4:,.0f}만원 ({eq/start:.2f}배) 최대낙폭 {mdd*100:.0f}%")

rng = np.random.default_rng(0)
def eval_sim(risk_usd, n=20000, max_trades=200):
    passed = blown = 0; tp = []
    for _ in range(n):
        eq = peak = 0; locked = False
        for k in range(1, max_trades + 1):
            eq += rng.choice(R) * risk_usd; peak = max(peak, eq)
            if not locked and peak >= 2600: locked = True
            if eq <= (100 if locked else peak - 2500): blown += 1; break
            if eq >= 3000: passed += 1; tp.append(k); break
    return passed / n, blown / n, np.median(tp) if tp else None
def funded_year(risk_usd, n=20000, trades=40):
    out = []; blown = 0
    for _ in range(n):
        eq = peak = 0; dead = False
        for _ in range(trades):
            eq += rng.choice(R) * risk_usd; peak = max(peak, eq)
            if eq <= peak - 2500: dead = True; break
        blown += dead; out.append(peak - 2500 if dead else eq)
    return np.mean(out), np.median(out), blown / n
print("\n[Apex 50K 근사: 목표 +$3,000, 트레일링 낙폭 $2,500]")
for rk in (250, 500, 750, 1000):
    p, b, t = eval_sim(rk); print(f" 리스크 ${rk}: 통과 {p*100:.0f}% 실패 {b*100:.0f}% 통과 중앙값 {t}거래")
print("[통과 후 실계좌 1년(40거래)]")
for rk in (250, 500, 750):
    m, md, b = funded_year(rk); print(f" 리스크 ${rk}: 평균 ${m:,.0f} 중앙값 ${md:,.0f} 1년 내 소멸 {b*100:.0f}%")
