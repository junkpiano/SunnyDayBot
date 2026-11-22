# SunnyDay Bot: Product Requirements Document

All features in one document, with their state on 2026-11-22. Each feature has its own
folder under [`features/`](features/): `spec.md` (what it does, with numbered acceptance
criteria) and `plan.md` (how it is built). This document holds what is product-wide; the
[TRD](TRD.md) holds the shared technical parts.

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

| Feature | IDs | State | Spec | Plan |
|---|---|---|---|---|
| NAV estimate and display | NAV-1–4 | Built | [spec](features/nav-estimate/spec.md) | [plan](features/nav-estimate/plan.md) |
| Estimation rules | EST-1–8 | Built | [spec](features/estimation/spec.md) | [plan](features/estimation/plan.md) |
| Fund coverage (34 funds) | FUND-1–3 | Built | [spec](features/funds/spec.md) | [plan](features/funds/plan.md) |
| Accuracy and drawdown | ACC-1–3 | Built | [spec](features/accuracy/spec.md) | [plan](features/accuracy/plan.md) |
| Factor attribution | ATTR-1–6 | Built | [spec](features/attribution/spec.md) | [plan](features/attribution/plan.md) |
| History: sparklines and charts | HIST-1–10 | Built | [spec](features/history/spec.md) | [plan](features/history/plan.md) |
| Favorites | FAV-1–3 | Built | [spec](features/favorites/spec.md) | [plan](features/favorites/plan.md) |
| Data freshness | FRESH-1 | Built | [spec](features/freshness/spec.md) | [plan](features/freshness/plan.md) |
| Markets outlook | MKT-1–3 | Built | [spec](features/markets/spec.md) | [plan](features/markets/plan.md) |
| Operations | OPS-1–3 | Built | [spec](features/operations/spec.md) | [plan](features/operations/plan.md) |
| Push / webhook notifications | R-1 | Roadmap | — | — |
| Custom dip alerts | R-2 | Roadmap | — | — |

Each acceptance criterion has an ID: the feature's prefix and a number. Tests are named after
them where they can be, and commits cite them (`Covers HIST-7.`).

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
- Accounts, tracking or any personal data.

Each feature's own non-goals are in its `spec.md`.

## 7. Development

- **New feature:** create `features/<feature>/spec.md` with new IDs and agree it before
  building. Then write `plan.md`, build against the IDs, and add the feature to the table
  above.
- **Change to a feature:** update its `spec.md` first, then its code, tests and `plan.md`.
- **Test:** `cargo test --release`. CI and builds are described in [TRD §5](TRD.md#5-build-test-and-ci).
- **Delivery:** changes land on `main`; the site follows on the next scheduled run.
