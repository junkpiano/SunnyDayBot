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

- Five years of daily closes (`HISTORY_YEARS`); Prophet with default settings is trained on the
  latest 180 (`TRAIN_DAYS`) and predicts 30 days (`FORECAST_DAYS`).
- The range is Prophet's interval, or ±1.96 standard deviations of the training closes when
  Prophet gives none.
- The chart (1280×720) shows the closes, the forecast and its range, and today's price marked.
- `main()` skips a market whose data or forecast fails and goes on.

## Tests

- `classifies_markets`: BUY, HOLD and SELL around the 2% thresholds (MKT-2).
- MKT-1 and MKT-3 depend on live data and the model; they are seen in a generated site and the run
  log.
