# Markets outlook — plan

Implements MKT-1 to MKT-3 of [spec.md](spec.md).

## Code

| Part | Where |
|---|---|
| Markets | `TICKERS`: `^IXIC` NASDAQ, `^GSPC` S&P500, `^N225` Nikkei225, `GC=F` Gold, `BTC-USD` Bitcoin |
| Forecast | `forecast_series()`: Prophet from `augurs` with the WASM Stan optimizer |
| Badge | `classify()`, threshold `THRESH` = 0.02 |
| Card | `asset_card()` |
| Chart | `render_chart()` with `plotters`, written as `site/<name>_forecast.png` |

## How it works

- The spec's settings are constants: `HISTORY_YEARS` = 5, `TRAIN_DAYS` = 180, `FORECAST_DAYS` = 30.
- The fallback range is computed by `stddev()` when `augurs` returns no interval.
- `main()` skips a market whose data or forecast fails and goes on.

## Tests

- `classifies_markets`: BUY, HOLD and SELL around the 2% thresholds (MKT-2).
- MKT-1 and MKT-3 depend on live data and the model; they are seen in a generated site and the run
  log.
