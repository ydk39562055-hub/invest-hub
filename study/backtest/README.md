# 백테스트 폴더

- `RESULTS.md` — 1차~6차 결과와 전체 결론
- `ict_backtest.py` — 스윕→MSS/CISD 전략 엔진 (필터, SMT, 1분 체결 판정 포함)
- `strategy_b_backtest.py` — 5분 FVG in 15분 FVG 전략 (미래 참조 제거 버전) + RR/트레일 스윕
- `jadecap_backtest.py` — JadeCap 1H SFP + 5분 FVG 전략 + 개선안 (`--improve`, `--long1h`)
- `jadecap_money_sim.py` — 100만 원 복리, 프롭펌 몬테카를로
- `logs/` — 각 실행의 원시 출력

## 데이터 (저장소에 없음, 용량 때문)

| 파일 | 출처 | 만드는 법 |
|---|---|---|
| `NQ_databento_5m.csv` | github.com/prashanthaitha24/nq-strategy-b-bot `data/nq_databento_5min.csv` | 그대로 복사 |
| `OANDA_NAS_5m.csv`, `OANDA_SPX_5m.csv` (+1m) | github.com/FutureSharks/financial-data `pyfinancialdata/data/currencies/oanda/{NAS100_USD,SPX500_USD}` | 연도별 CSV 합쳐 UTC 인덱스로 저장, 5분 리샘플 |
| `SPXUSD_5m_2010_2018.csv` (+1m) | 같은 저장소 `data/stocks/histdata/SPXUSD` | `;` 구분, EST 고정(UTC-5) → UTC 변환 |
| `SPX500_5m.csv`, `SPX500_1m.csv`, `NAS100_1m.csv` | github.com/getdata-finance/{spx500-5m,spx500-1m,nas100-1m}-ohlcv-index-historical-data | 그대로 복사 |

CSV 형식: `datetime(ISO, tz 포함),open,high,low,close,volume`

## 실행 예
```
python ict_backtest.py SPX500_5m.csv --m1 SPX500_1m.csv --compare
python strategy_b_backtest.py OANDA_NAS_5m.csv --corr OANDA_SPX_5m.csv
python strategy_b_backtest.py NQ_databento_5m.csv --rr
python jadecap_backtest.py NQ_databento_5m.csv
python jadecap_backtest.py OANDA_NAS_5m.csv --improve --long1h
python jadecap_money_sim.py NQ_databento_5m.csv
```
