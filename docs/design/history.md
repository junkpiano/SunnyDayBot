# Design: fund history and charts

Implements [spec/history.md](../../spec/history.md) (HIST-1 to HIST-10).

## Goal

Each fund card gets a 30-day sparkline. Tapping a card opens a chart of the fund's NAV since
inception, the estimates scored against it, and its drawdown. The first page load stays small,
and the site stays a static set of files built by one binary run.

## Data flow

```
NAV CSV (since inception) ─┐
                           ├─ history_file() ─> site/history/<fund_id>.json ── fetched on open ─> chart
history.json (estimates) ──┘
NAV CSV, last 30 days ───────── sparkline() ──> inline <svg> in index.html
```

Each run already downloads every fund's full NAV CSV and the published `history.json`
(see [operations](../../spec/operations.md)). Both inputs are complete, so the history files
are rebuilt from scratch on every run. Nothing is appended, and nothing has to be read back.

## History file

`site/history/<fund_id>.json`:

```json
{"fields": ["date", "actual_nav", "estimated_nav", "error_jpy", "error_pct", "drawdown_pct"],
 "rows": [["2026-11-20", 35574.0, 35558.0, -16.0, -0.045, -8.8], ...]}
```

- There is one row per published NAV, oldest first. Rows are arrays and the field names appear
  once, which keeps the file small.
- The estimate columns are null where no estimate was scored against that NAV. An estimate is
  matched to a NAV through its `history.json` entry: the entry's `base` is the NAV before it,
  so the NAV it was scored against is the first CSV row after `base`. Pending and unscored
  entries (`estimate: null`) are skipped.
- error_jpy = estimate − actual, rounded to 2 decimals. error_pct = (estimate ÷ actual − 1) ×
  100, rounded to 3 decimals.
- drawdown_pct = (NAV ÷ highest NAV up to that day − 1) × 100, rounded to 2 decimals.
- **`fund_id`** comes from the NAV CSV address, so the fund list needs no new field. It is the
  file name without `.csv` for MUFG and Rakuten (`253425`, `chart_100086`), and the `isinCd`
  value for the Investment Trusts Association (`JP90C000J569`).

Measured on 2026-11-22 across all 34 funds: the largest file is 171 KB raw and 31 KB gzipped
(三菱UFJ 純金ファンド, the longest history). HIST-9's limit is 50 KB gzipped.

## Keeping estimates

The estimates only exist in `history.json`. That file used to keep the latest 120 entries per
fund, which also set the window the card's estimate error averaged over. These are now two
constants:

- `HISTORY_KEEP = 2500`: entries kept per fund, about ten years of NAVs.
- `SCORE_WINDOW = 120`: the latest entries the card's average error is computed over.

Growth: one scored entry is about 70 bytes, so 34 funds × ~245 NAVs a year ≈ 600 KB a year
added to `history.json`. It is read and written once per run.

### Considered and rejected: read each history file back from the live site

This would have left `history.json` at 120 entries per fund, with each history file appended to
from its published copy. In a trial run GitHub Pages answered 503 for 9 of the 34 files.

- A file that can't be read can't be safely rewritten, so it is left out of that run's publish.
- The next run then gets a 404, treats the file as new, and loses every estimate older than
  `history.json`'s 120.

It also costs 34 more requests per run. Keeping the estimates in `history.json` puts them all
behind the one read that already stops the run on failure (OPS-2).

## Sparkline (HIST-1)

`sparkline()` renders the last 30 days of NAVs, from the NAV CSV, as a `<polyline>` in a 100×30
viewBox, stretched to the card's width (`preserveAspectRatio="none"`,
`vector-effect: non-scaling-stroke`). Its class is `up`, `down` or `flat` by the move from the
first point to the last, so it takes the card's colours. It is drawn server-side, so it needs no
script and no extra request.

## Chart (HIST-2 to HIST-6, HIST-10)

The chart is plain script in `src/template.html`, with no library:

- **Open and close:** one delegated `click` handler covers the ★ button, the chart's controls
  (which never toggle the card) and the card itself. Cards get `tabIndex = 0`, and Enter clicks
  them.
- **Fetching:** the first open fetches `history/<fund_id>.json` into a per-page cache. A failed
  fetch leaves the cache empty, so opening the card again retries, and the card says "History
  unavailable."
- **Drawing:** the chart is a 320×140 viewBox SVG, rebuilt on each control change.
  - The range filters rows by calendar days back from the latest NAV: 1M 31, 3M 92, 1Y 366,
    Max all.
  - NAV mode draws one path for the NAV and a dot for each estimate. The dots stay visible
    when estimates are sparse or isolated, where a broken line would vanish.
  - Drawdown mode draws one path, with 0% as the top of the scale.
- **Scrubbing:** `pointermove` maps the pointer's x to the nearest row and shows that day's
  figures; the latest day is shown first. `touch-action: pan-y` lets a horizontal drag scrub
  while a vertical swipe still scrolls the page.
- **Favorites:** the favorites section shows copies of cards. Copies are made without an open
  chart, so a starred card's chart isn't duplicated in a stale state.

## Cost

| | Before | After |
|---|---|---|
| `index.html` | 28.7 KB (5.1 KB gzipped) | 45.4 KB (9.0 KB gzipped) |
| Requests on first load | page + 5 PNGs | same |
| Per opened card | — | one history file, 0.5–31 KB gzipped |
| Run time | — | 34 small file writes; no extra requests |

## Testing

- Unit tests in `src/main.rs`:
  - `writes_fund_history` (HIST-7: matching, errors, drawdown);
  - `keeps_estimates_past_the_score_window` (HIST-8);
  - `draws_sparklines` (HIST-1);
  - `names_funds_by_code`.
  
  CI runs them.
- The page script was checked once by hand, on 2026-11-22, with headless Chromium (Playwright)
  against a locally generated site. CI doesn't run it. The check covered:
  - no fetch on load, one fetch on open and none on reopen;
  - 3M and NAV selected first; mode and range switching;
  - scrubbing;
  - ★ not toggling the chart;
  - a failed fetch.

## Limits

- `history.json` grows by about 600 KB a year until it reaches 2,500 entries per fund. Past
  that, the oldest estimates drop out of the history files too. Raise `HISTORY_KEEP`, or split
  `history.json` per fund, if it gets slow.
- Estimates exist only from early November 2026, when the bot started recording them.
