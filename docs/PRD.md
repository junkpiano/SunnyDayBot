# SunnyDay Bot: Product Requirements Document

All features in one document, with their state on 2026-11-22. This is where the requirements
live: each has an ID (§3), and changes, tests and commit messages cite those IDs. The
[TRD](TRD.md) describes how the product is built.

## 1. Overview

SunnyDay Bot is a lightweight, mobile-first web page. It estimates today's Net Asset Value
(NAV / 基準価額) of Japanese mutual funds before the asset managers publish it.

Japanese funds that invest abroad price their NAV for day D at the foreign closes of the night
before, converted at D's USD/JPY TTM. Investors usually wait until the evening to see it.
SunnyDay Bot estimates it every morning and shows how accurate past estimates were. It also
shows each fund's high, drawdown and history, and a 30-day outlook for the underlying markets.

Site: <https://junkpiano.github.io/SunnyDayBot/>

## 2. Users

- **Japanese retail index investors** holding popular NISA funds (eMAXIS Slim All Country,
  eMAXIS Slim S&P 500, Rakuten SCHD and similar). They want to see the day's move before the
  NAV is out.
- **DIY investors** who want to see how accurate the estimates are and where a fund stands
  against its high, on a clean mobile page.

## 3. Features

| Feature | IDs | State |
|---|---|---|
| NAV estimate and display | NAV-1–4 | Built |
| Estimation rules | EST-1–8 | Built |
| Fund coverage (34 funds) | FUND-1–3 | Built |
| Accuracy and drawdown | ACC-1–3 | Built |
| Factor attribution | ATTR-1–6 | Built |
| History: sparklines and charts | HIST-1–10 | Built |
| Favorites | FAV-1–3 | Built |
| Data freshness | FRESH-1 | Built |
| Markets outlook | MKT-1–3 | Built |
| Operations | OPS-1–3 | Built |
| Push / webhook notifications | — | Roadmap |
| Custom dip alerts | — | Roadmap |

Each requirement has an ID: the feature's prefix and a number. Each is a checkable acceptance
criterion; tests are named after them where they can be, and commits cite them (`Covers
HIST-7.`).

### 3.1 NAV estimate and display (NAV)

- **NAV-1** Funds are shown in their groups (MSCI, S&P500, NASDAQ100, FANG+, SOX, SCHD, Gold,
  Gold Plus). Each card shows the latest official NAV and its date (`NAV 11/20`) and the
  estimate of today's NAV (`11/23 NAV estimate`, the next weekday on a weekend), in 円.
- **NAV-2** The estimate's % change is shown with a weather icon: ☀️ rise, ☔ fall, ☁️ no
  change.
- **NAV-3** A rise is coloured green, a fall red, no change grey.
- **NAV-4** Once today's NAV is out, the card shows that NAV instead, with its change in 円 and %
  from the previous NAV, and says the next estimate comes with the 07:17 update.

### 3.2 Estimation rules (EST)

- **EST-1** The target date is today in Japan; on Saturday and Sunday it is Monday.
- **EST-2** An index's change runs from the last close before the NAV date to the last close
  before the target date; later closes belong to the next NAV.
- **EST-3** A leg's weight multiplies its move (2x, 3x funds); two legs add their moves.
- **EST-4** An unhedged fund converts from the NAV date's USD TTM to the target date's TTM once
  it is out, before that to the latest hourly USD/JPY.
- **EST-5** A hedged fund doesn't move with USD/JPY.
- **EST-6** A fund holding a Tokyo ETF follows that ETF once it has traded after the NAV date.
- **EST-7** A published NAV is scored against the last estimate made before 15:30 JST on its
  date.
- **EST-8** The USD TTM is read from MUFG's page for the exact date, or not at all.

### 3.3 Fund coverage (FUND)

- **FUND-1** All 34 funds in [TRD appendix A](TRD.md#appendix-a-funds) have a card, in their group, in
  that order. They include leveraged (2x, 3x), yen-hedged and gold-plus funds, each defined by
  its NAV CSV (fund company or 投資信託協会), the tickers it follows with weights, its hedge,
  and a Tokyo ETF where it has one.
- **FUND-2** NAV CSVs are read with both `2026/11/02` and `2026年11月02日` dates; other rows are
  skipped.
- **FUND-3** A fund whose data can't be fetched is left off that update; the others still show.

### 3.4 Accuracy and drawdown (ACC)

- **ACC-1** The card shows the highest NAV since inception, its date and today's drawdown from it
  (`High since inception 39,019円 on 2026-08-14 · -8.8%`).
- **ACC-2** Badges: `🎉 New high (estimate)` when the estimate passes the high; `🎉 High on
  <date>` or `🎉 High today` when the published NAV is the high.
- **ACC-3** The card shows the average estimate error in 円 and % over the latest 120 scored
  NAVs, and the latest comparison (`Estimate error: avg 70円 (0.19%) over 12 NAVs · after 11/19:
  35,558 vs actual 35,574`).

### 3.5 Factor attribution (ATTR)

- **ATTR-1** Every estimating card has a line under its % change saying what moved it; a
  published card has none.
- **ATTR-2** A one-index fund shows that index's change (`S&P500 -3.70%`); a leveraged fund adds
  its leverage (`NASDAQ100 -1.50% ×3`).
- **ATTR-3** A fund with several indexes shows each (`MSCI ACWI +0.40% · Gold +1.10%`).
- **ATTR-4** An unhedged fund shows the USD/JPY change the estimate used; a hedged fund shows
  `USD/JPY hedged`.
- **ATTR-5** The parts multiply to the card's change: (1 + index) × (1 + USD/JPY) − 1.
- **ATTR-6** A fund priced from its Tokyo ETF shows the ETF's yen change alone (`1540 (yen)
  +0.85%`).

### 3.6 History: sparklines and charts (HIST)

- **HIST-1** Each fund card shows a 30-day NAV sparkline, coloured by the move.
- **HIST-2** No history is fetched on page load; a fund's history is fetched once, when its card
  is first opened.
- **HIST-3** Tapping a card opens its chart and tapping again closes it; ★ and the chart's own
  controls don't.
- **HIST-4** Timeframes 1M, 3M, 1Y and Max, and NAV or Drawdown mode; 3M and NAV first.
- **HIST-5** NAV mode shows published NAVs as a line and past estimates as dots; Drawdown mode
  shows the % below the running high.
- **HIST-6** A tooltip follows the finger or pointer: date, NAV, estimate, error, drawdown.
- **HIST-7** History goes back to each fund's inception, from its NAV CSV, with the fields date,
  actual_nav, estimated_nav, error_jpy, error_pct and drawdown_pct.
- **HIST-8** Old estimates stay in the history after they leave the 120-NAV error average.
- **HIST-9** Each fund's history is under 50 KB gzipped.
- **HIST-10** If a fund's history can't be fetched, its card says so and the rest of the page
  works.

### 3.7 Favorites (FAV)

- **FAV-1** Every card, fund or market, has a ☆ that turns to ★ when starred.
- **FAV-2** Starred cards appear in a "★ Favorites" section at the top, hidden when empty.
- **FAV-3** Favorites are kept in this browser (localStorage), with no account, and survive a
  reload.

### 3.8 Data freshness (FRESH)

- **FRESH-1** The header shows when the page was built, in JST (`Updated 2026-11-22 20:26 JST`).

### 3.9 Markets outlook (MKT)

- **MKT-1** NASDAQ, S&P 500, Nikkei 225, gold and Bitcoin each have a card: current price,
  30-day forecast (Prophet, trained on the latest 180 closes), change, range, badge and a
  five-year chart with the forecast.
- **MKT-2** The badge is BUY when the price is more than 2% below the forecast, SELL when more
  than 2% above, HOLD otherwise.
- **MKT-3** A market whose data or forecast fails is left out; the others still show. The page
  states the figures are mechanical and not investment advice.

### 3.10 Operations (OPS)

- **OPS-1** Every pull request and push to `main` is tested in CI; `main` also builds the binary
  that the site is generated with.
- **OPS-2** If the published estimate history can't be read, the update stops rather than
  publish an empty one.
- **OPS-3** Up to 2,500 estimates are kept per fund.

## 4. Non-functional requirements

| ID | Requirement | State |
|---|---|---|
| NFR-1 | **Static site** on GitHub Pages, rebuilt on a schedule (07:17, 13:17 and hourly 19:17–23:17 JST) | Met. Built by one binary run and published to `gh-pages` |
| NFR-2 | **Fast first load**: only cards and sparklines at first, history on demand | Met. `index.html` is 9 KB gzipped; no history is fetched on load |
| NFR-3 | **History payload** under 50 KB gzipped per fund | Met. Largest 31 KB (三菱UFJ 純金ファンド) |
| NFR-4 | **Mobile and dark mode**: mobile viewports, light and dark themes following the system, tabular numerals, touch-scrubbable chart | Met. Touch scrubbing not yet tried on a real phone |
| NFR-5 | **Accuracy**: MAPE under 0.25% across the major global and US equity index funds | Met. 0.080% over 122 NAVs across 10 funds (estimates for NAVs after 2026-11-02 to 11-19). Gold funds are higher (up to about 0.7%) |
| NFR-6 | **No personal data**: no accounts, nothing sent from the browser | Met. Favorites stay in localStorage |

## 5. Roadmap

- **R-1 Push / webhook notifications:** an optional daily summary via Telegram or Web Push when an
  estimate moves more than a chosen threshold (for example ±2.0%).
- **R-2 Custom dip alerts:** a notification when a favorited fund crosses a drawdown threshold from
  its high (for example −10% or −15%).

## 6. Not doing

- Investment advice: every figure is a mechanical estimate, and the page says so.
- Splitting a published NAV's change into parts: the fund companies don't publish what moved
  it.
- Storing the attribution parts in the history files or charting them: only today's estimate
  is split.
- Estimates from before the bot began recording them (early November 2026): they were never
  made.
- A separate CSV backfill: the NAV CSVs already cover every NAV since inception.
- History charts for the market cards: they already have the forecast PNG.
- A charting library: the chart is an inline SVG drawn by a few lines of script.
- Describing one particular host's setup in the repository.

## 7. Development

- A big change starts with its requirements here, with new IDs, agreed before it is built; a
  design doc in `docs/design/` records how a non-trivial one is built.
- Test: `cargo test --release`. CI and builds: [TRD §7](TRD.md#7-build-test-and-ci).
- Changes land on `main`; the site follows on the next scheduled run.
