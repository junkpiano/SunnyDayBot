# SunnyDayBot

Estimates mutual fund NAVs before they are published, with market forecasts, as a static site.

## What It Does
- Fetches historical prices for NASDAQ, S&P 500, Nikkei 225, gold and Bitcoin.
- Trains a Prophet model on the most recent window of data.
- Forecasts the next 30 days and classifies the current price as BUY/SELL/HOLD.
- Estimates today's NAV of mutual funds (MSCI, S&P500, NASDAQ100, FANG+, SOX, SCHD, gold and gold-plus funds, including leveraged and yen-hedged ones), marks new highs since inception, and scores past estimates against the published NAVs (`site/history.json`, read back from the live site on each run).
- Renders charts and writes the site to `site/`.

## Requirements
- Rust toolchain (stable)
- `pkg-config` and fontconfig headers for the charts (`libfontconfig1-dev` on Debian/Ubuntu)

## Local Run
```bash
cargo run --release
```

Artifacts:
- `site/index.html` (from `src/template.html`)
- `site/*_forecast.png`

## CI
`.github/workflows/build.yml` runs the unit tests, and on `main` also builds an arm64 binary as a run artifact.

## Configuration
Adjust these constants in `src/main.rs`:
- `TICKERS` for tracked indices
- `FUNDS` for mutual funds (NAV CSV, tracked tickers with weights, yen hedge)
- `HISTORY_YEARS`, `TRAIN_DAYS`, `FORECAST_DAYS`
- `THRESH` for BUY/SELL thresholds

