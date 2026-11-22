# Product Requirements Document (PRD): SunnyDay Bot

## 1. Overview & Background

SunnyDay Bot is a lightweight, mobile-first web application designed to estimate the next-day Net Asset Value (NAV / 基準価額) of Japanese mutual funds (investment trusts) before official daily pricing is published by asset management companies.

Japanese mutual funds investing in overseas assets calculate their daily NAV using the previous night's foreign market closing prices and the morning's USD/JPY TTM exchange rate. Retail investors often wait until evening for official NAV updates. SunnyDay Bot bridges this latency gap by providing highly accurate, automated estimates alongside historical tracking metrics and chronological performance visualizations.

## 2. Target Audience & User Personas

- **Japanese Retail Index Investors:** Individual investors holding popular NISA-eligible mutual funds (e.g., eMAXIS Slim All Country, eMAXIS Slim S&P 500, Rakuten SCHD) who want real-time visibility into daily portfolio fluctuations.
- **Tech-Savvy DIY Investors:** Users who value transparent estimation accuracy (tracking historical estimate errors), drawdown context, and clean, distraction-free mobile interfaces.

## 3. Core Features & Functional Requirements

One file per feature:

- [nav-estimate](nav-estimate.md)
- [accuracy-and-drawdown](accuracy-and-drawdown.md)
- [history](history.md)
- [favorites](favorites.md)
- [data-freshness](data-freshness.md)
- [markets](markets.md)

How it works:

- [funds](funds.md): the funds and what each holds
- [estimation](estimation.md): how a NAV is estimated and scored
- [operations](operations.md): build, schedule, publishing, history, data sources

## 4. Non-Functional Requirements

- **Architecture & Hosting:** Static frontend hosted on GitHub Pages (junkpiano.github.io), updated via automated scheduled CI/CD pipelines (e.g., GitHub Actions) that fetch index/FX closing data, compute estimates, append historical records, and deploy static JSON/HTML assets.
- **Lazy-Loaded History Payloads:** Keep initial page load instantaneous by loading only summary card data + 30-day sparkline points on initial render; fetch full multi-year `history/<fund_id>.json` files lazily on card expansion (target payload < 50 KB gzipped per fund).
- **UI/UX & Accessibility:** Native dark-mode theme optimized for mobile viewports (iOS Safari/Chrome) with touch-friendly scrubbable chart tooltips and high-contrast tabular numerals.
- **Accuracy Target:** Maintain a mean absolute percentage error (MAPE) under 0.25% across major global and US equity index funds.

## 5. Future Roadmap (Post-MVP Ideas)

- **Factor Attribution Breakdown:** Show how much of the estimated NAV change comes from underlying index movement vs. USD/JPY exchange rate fluctuation.
- **Push / Webhook Notifications:** Optional daily summary alerts via Telegram or Web Push when estimates exceed a user-defined volatility threshold (e.g., ±2.0%).
- **Custom Dip Alerts:** Notify users when a favorited fund crosses a target drawdown threshold from its All-Time High (e.g., -10% or -15%).

## Development

- Test: `cargo test --release` (needs `pkg-config` and fontconfig headers).
- CI: `.github/workflows/build.yml` on GitHub runs the tests on pull requests and pushes to
  `main`, and on `main` keeps the arm64 binary.
- Changes land on `main`; the published site follows on the next scheduled run.
