# SunnyDayBot

Estimates mutual fund NAVs before they are published, with market forecasts, as a static site.

## What It Does
SunnyDay Bot is a lightweight, mobile-first web page: <https://junkpiano.github.io/SunnyDayBot/>.

Japanese funds that invest abroad price their NAV for day D at the foreign closes of the night
before, converted at D's USD/JPY TTM, and investors usually wait until the evening to see it.
SunnyDay Bot estimates it every morning, shows how accurate past estimates were, each fund's
high, drawdown and history, and a 30-day outlook for the underlying markets.

It is for Japanese retail index investors holding popular NISA funds (eMAXIS Slim All Country,
eMAXIS Slim S&P 500, Rakuten SCHD and similar) who want the day's move before the NAV is out,
and for DIY investors who want to see how accurate the estimates are.

## Documents
Each feature has a spec (what it does, with numbered acceptance criteria) and a plan (how it is
built). Tests are named after the IDs where they can be, and commits cite them (`Covers HIST-7.`).

| Feature | IDs | State | Spec | Plan |
|---|---|---|---|---|
| NAV estimate and display | NAV-1–4, NFR-4 | Built | [spec](docs/features/nav-estimate/spec.md) | [plan](docs/features/nav-estimate/plan.md) |
| Estimation rules | EST-1–8, NFR-5 | Built | [spec](docs/features/estimation/spec.md) | [plan](docs/features/estimation/plan.md) |
| Fund coverage (34 funds) | FUND-1–3 | Built | [spec](docs/features/funds/spec.md) | [plan](docs/features/funds/plan.md) |
| Accuracy and drawdown | ACC-1–3 | Built | [spec](docs/features/accuracy/spec.md) | [plan](docs/features/accuracy/plan.md) |
| Factor attribution | ATTR-1–6 | Built | [spec](docs/features/attribution/spec.md) | [plan](docs/features/attribution/plan.md) |
| History: sparklines and charts | HIST-1–10, NFR-2–3 | Built | [spec](docs/features/history/spec.md) | [plan](docs/features/history/plan.md) |
| Favorites | FAV-1–3, NFR-6 | Built | [spec](docs/features/favorites/spec.md) | [plan](docs/features/favorites/plan.md) |
| Data freshness | FRESH-1 | Built | [spec](docs/features/freshness/spec.md) | [plan](docs/features/freshness/plan.md) |
| Markets outlook | MKT-1–3 | Built | [spec](docs/features/markets/spec.md) | [plan](docs/features/markets/plan.md) |
| Operations | OPS-1–3, NFR-1 | Built | [spec](docs/features/operations/spec.md) | [plan](docs/features/operations/plan.md) |

## Roadmap
- **R-1 Push / webhook notifications:** an optional daily summary via Telegram or Web Push when an
  estimate moves more than a chosen threshold (for example ±2.0%).
- **R-2 Custom dip alerts:** a notification when a favorited fund crosses a drawdown threshold from
  its high (for example −10% or −15%).

## Not doing
- Investment advice: every figure is a mechanical estimate, and the page says so.
- Accounts, tracking or any personal data.

Each feature's own non-goals are in its spec.

## Development
The documents are the product; `src/` is one implementation of them and can be replaced. Each
spec says what the product does and the contracts it must keep (data sources, published file
formats), with no language, framework or library in it, so the product can be rebuilt in any
stack and checked against the same acceptance IDs. Each plan says how the current
implementation does it.

- **New feature:** create `docs/features/<feature>/spec.md` with new IDs and agree it before
  building. Then write its `plan.md`, build against the IDs, and add it to the table above.
- **Change to a feature:** update its `spec.md` first, then its code, tests and `plan.md`.
- **Delivery:** changes land on `main`; the site follows on the next scheduled run.

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
- `site/history/<fund>.json`, each fund's NAVs, estimates and drawdown, fetched when its card is opened

## CI
`.github/workflows/build.yml` runs `cargo test --release` on every pull request and push to `main`, and on `main` also builds the arm64 binary as a run artifact. Building needs rustc 1.95 or newer.

## Configuration
Adjust these constants in `src/main.rs`:
- `TICKERS` for tracked indices
- `FUNDS` for mutual funds (NAV CSV, tracked tickers with weights, yen hedge)
- `HISTORY_YEARS`, `TRAIN_DAYS`, `FORECAST_DAYS`
- `THRESH` for BUY/SELL thresholds

