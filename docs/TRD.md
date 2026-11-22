# SunnyDay Bot: Technical Requirements Document

The technical parts every feature shares. Each feature's own design is its
[`plan.md`](PRD.md#3-features); requirement IDs are the features' `spec.md`.

## 1. Architecture

```
            ┌──────────── scheduled host (cron, arm64) ─────────────┐
Yahoo  ───> │                                                        │
MUFG TTM ─> │  sunnydaybot (one Rust binary)  ──>  site/  ──git push─┼─> gh-pages ─> GitHub Pages ─> browser
NAV CSVs ─> │        ▲                                               │                    │
            │        └── history.json read back from the live site ──┼────────────────────┘
            └────────────────────────────────────────────────────────┘
```

- **One process:** a single run fetches every input, computes estimates and forecasts, and
  writes a complete static site. There is no server, database or browser-side API key.
- **State:** the only state carried between runs is `history.json`. Each run reads it back
  from the live site and publishes it again.
- **Browser script:** the page's script is plain JavaScript inlined in `index.html`, with no
  framework and no library. It does favorites and charts.

## 2. Code map

All the code is in `src/main.rs`, and the page template with its script in `src/template.html`.
Each feature's `plan.md` names its functions:

| Feature | Main functions |
|---|---|
| [NAV estimate](features/nav-estimate/plan.md) | `build_html`, `fund_card`, `pct_class`, `weather`, `fmt_num` |
| [Estimation](features/estimation/plan.md) | `estimate_funds`, `nav_target`, `change_between`, `change_since`, `change_since_time`, `usd_ttm`, `parse_ttm`, `record` |
| [Funds](features/funds/plan.md) | `FUNDS`, `fetch_nav` |
| [Accuracy](features/accuracy/plan.md) | `score`, `accuracy_line` |
| [Attribution](features/attribution/plan.md) | `Attribution`, `leg_name` |
| [History](features/history/plan.md) | `history_file`, `archive`, `fund_id`, `sparkline`; chart script |
| [Favorites](features/favorites/plan.md) | favorites script |
| [Freshness](features/freshness/plan.md) | `build_html` |
| [Markets](features/markets/plan.md) | `forecast_series`, `classify`, `asset_card`, `render_chart` |
| [Operations](features/operations/plan.md) | `main`, `load_history`, `fetch_series`, `fetch_chart` |

Dependencies are `augurs` (Prophet), `plotters` (PNG charts), `reqwest` (blocking HTTP),
`serde_json`, `chrono`, `anyhow` and `urlencoding`.

## 3. Shared data

### 3.1 Sources

| Data | Source | Used for |
|---|---|---|
| Index, ETF, gold, Bitcoin closes; USD/JPY daily and hourly | Yahoo Finance chart API | Estimates, forecasts |
| Fund NAVs since inception | Fund company CSVs (MUFG, Rakuten) or the Investment Trusts Association library | Latest NAV, high, history, scoring |
| USD TTM | MUFG (`murc-kawasesouba.jp`) | The currency part of estimates |

The fund list, with each fund's CSV, tickers, weights, hedge and Tokyo ETF, is the `FUNDS`
constant; [funds/spec.md](features/funds/spec.md) has it as a table.

### 3.2 `history.json` (state between runs)

```json
{"<fund name>": [
  {"base": "2026-11-02", "runs": [["2026-11-03T22:17:00Z", 37306.2], ...], "estimate": null, "actual": null},
  {"base": "2026-10-30", "estimate": 37100.0, "actual": 37090.0}
]}
```

- `base` is the NAV date the estimate starts from. While the next NAV is pending, every run
  appends `[time, estimate]` to `runs`.
- Once that NAV is out, `runs` is replaced by the chosen `estimate` and the `actual` NAV.
- Up to 2,500 entries are kept per fund (`HISTORY_KEEP`). The card's error averages the latest
  120 (`SCORE_WINDOW`).
- Estimates are kept here rather than read back from each published history file: in a trial,
  GitHub Pages answered 503 for 9 of 34 such reads, and a file that can't be read can't be
  rewritten safely.

The per-fund history files are described in [history/plan.md](features/history/plan.md).

## 4. Page

`src/template.html` is filled in with the cards and the update time.

- **Styles:** CSS variables for the light and dark themes (`prefers-color-scheme`), and
  `font-feature-settings: "tnum"` for tabular numerals; cards in a responsive grid.
- **Script:** plain JavaScript, inline, no framework or library. One delegated click handler
  serves favorites and charts ([favorites](features/favorites/plan.md),
  [history](features/history/plan.md)).

## 5. Build, test and CI

- **Toolchain:** Rust stable; the locked dependencies need rustc 1.95 or newer. Building needs
  `pkg-config` and the fontconfig headers.
- **Tests:** `cargo test --release`. Each feature's `plan.md` lists its tests and the IDs they
  cover.
- **CI:** `.github/workflows/build.yml` on `ubuntu-24.04-arm` tests every pull request and push
  to `main`, and on `main` keeps the arm64 binary as a run artifact for 90 days.
- **Not in CI:** the page script, checked by hand in headless Chromium.

## 6. Performance (NFR-2, NFR-3)

| | Measured 2026-11-22 |
|---|---|
| `index.html` | 45 KB (9 KB gzipped) |
| Requests on first load | the page and 5 market PNGs |
| History file per fund | 0.5–31 KB gzipped |
| `history.json` growth | about 600 KB a year until 2,500 entries per fund |

## 7. Security and privacy (NFR-6)

- **No accounts and no tracking.** The browser sends nothing. Favorites stay in localStorage.
- **Read-only inputs.** The bot only reads public sources.
- **Credentials:** the publishing host's GitHub credentials are its only secret, and they
  never enter the repository.
- **Repository contents:** the repository describes the setup, not any one host.
