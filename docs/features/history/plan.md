# History: sparklines and charts — plan

Implements HIST-1 to HIST-10 of [spec.md](spec.md).

## Data flow

```
NAV CSV (since inception) ─┐
                           ├─ history_file() ─> site/history/<fund_id>.json ── fetched on open ─> chart
history.json (estimates) ──┘
NAV CSV, last 30 days ───────── sparkline() ──> inline <svg> in index.html
```

Each run already has every fund's full NAV CSV and the estimates in `history.json`, so the
history files are rebuilt from scratch each run; nothing is appended or read back.

## History file

`site/history/<fund_id>.json`:

```json
{"fields": ["date", "actual_nav", "estimated_nav", "error_jpy", "error_pct", "drawdown_pct"],
 "rows": [["2026-11-20", 35574.0, 35558.0, -16.0, -0.045, -8.8], ...]}
```

- One row per published NAV, oldest first; arrays with the field names once, to keep it small.
- An estimate is matched to a NAV through its `history.json` entry: the entry's `base` is the NAV
  before it, so its NAV is the first CSV row after `base`. Unscored entries are skipped, and those
  rows have nulls.
- error_jpy = estimate − actual (2 decimals); error_pct = (estimate ÷ actual − 1) × 100 (3
  decimals); drawdown_pct = (NAV ÷ highest NAV so far − 1) × 100 (2 decimals).
- `fund_id()`: the CSV's file name without `.csv` (`253425`, `chart_100086`), or the `isinCd`
  value (`JP90C000J569`).
- **Keeping estimates (HIST-8):** `history.json` keeps `HISTORY_KEEP` = 2,500 entries per fund
  (about ten years), while the card's error uses the latest `SCORE_WINDOW` = 120. Reading each
  published history file back instead was tried and rejected: GitHub Pages answered 503 for 9 of
  34 reads, and a file that can't be read can't be rewritten safely.

## Sparkline (HIST-1)

`sparkline()` draws the last 30 days of NAVs as a `<polyline>` in a 100×30 viewBox stretched to
the card (`preserveAspectRatio="none"`, `vector-effect: non-scaling-stroke`). Its class is `up`,
`down` or `flat` by first-to-last move, so it takes the card's colours. No script, no request.

## Chart (HIST-2 to HIST-6, HIST-10)

Plain script in `src/template.html`, no library:

- One delegated click handler: ★, the chart's controls (never toggle the card), the card itself.
  Cards get `tabIndex = 0` and Enter clicks them.
- First open fetches `history/<fund_id>.json` into a per-page cache; a failed fetch leaves it
  empty (retried on the next open) and the card says "History unavailable."
- A 320×140 viewBox SVG, rebuilt on each control change. The range keeps rows within 31, 92 or
  366 days of the latest NAV, or all. NAV mode draws one path and a dot per estimate (dots stay
  visible where estimates are sparse); Drawdown mode draws one path with 0% at the top.
- `pointermove` maps x to the nearest row; `touch-action: pan-y` lets a horizontal drag scrub
  while a vertical swipe scrolls. The latest day is shown first.
- Favorites copies are made without an open chart.

## Cost (measured 2026-11-22)

| | Before | After |
|---|---|---|
| `index.html` | 28.7 KB (5.1 KB gzipped) | 45.4 KB (9.0 KB gzipped) |
| Per opened card | — | one file, 0.5–31 KB gzipped (largest: 三菱UFJ 純金ファンド) |
| `history.json` | — | grows about 600 KB a year until 2,500 entries per fund |

## Tests

| Test | Covers |
|---|---|
| `writes_fund_history` | HIST-7: matching, errors, drawdown |
| `keeps_estimates_past_the_score_window` | HIST-8 |
| `draws_sparklines` | HIST-1 |
| `names_funds_by_code` | `fund_id()` |

The page script isn't in CI. It was checked once by hand on 2026-11-22 in headless Chromium
(Playwright) against a locally built site: HIST-2, 3, 4, 6 and 10, with no script errors.

## Limits

- Once a fund passes 2,500 entries, its oldest estimates drop out of its history. Raise
  `HISTORY_KEEP` or split `history.json` per fund if it gets slow.
