# Chronological History & Time-Series Visualization

## Purpose

Show where each fund has been, and how close past estimates came, without slowing the first
page load.

## Behaviour

- **Card-Level Inline Sparklines:** Render a minimal, non-interactive 30-day SVG sparkline inside each fund card to show recent directional momentum at a glance without cluttering the layout.
- **Expandable Interactive Chart View:** Tapping a fund card expands an interactive time-series chart supporting selectable timeframes (1M, 3M, 1Y, Max).
- **Multi-Series & Drawdown Modes:**
  - **NAV Trend (Actual vs. Estimated):** Plot the official historical NAV alongside historical daily estimates so users can visually inspect tracking accuracy over time.
  - **Drawdown Curve:** Optional toggle displaying historical percentage decline from running All-Time Highs to contextualize market corrections.
- **Daily Automated Archiving & Backfill:**
  - Scheduled CI/CD jobs append daily reconciliation records (date, estimated_nav, actual_nav, error_jpy, error_pct, drawdown_pct) to static JSON files (`data/history/<fund_id>.json`).
  - Support one-time CSV backfilling from official asset manager / Investment Trusts Association (投資信託協会) historical records so multi-year charts work immediately.

How this is built (details in [the design doc](../docs/design/history.md)):

- **History file.** Each run writes `history/<fund_id>.json` next to `index.html` (the site root
  is what gets published, so `data/` is dropped from the path):
  `{"fields": ["date", "actual_nav", "estimated_nav", "error_jpy", "error_pct", "drawdown_pct"],
  "rows": [["2026-11-20", 35574, 35558, -16, -0.045, -8.8], ...]}`, one row per published NAV,
  oldest first. The estimate fields are null where no estimate was scored against that NAV.
  Error = estimate − actual.
- **Fund id.** The fund code in its NAV CSV address: `253425` (MUFG), `chart_100086` (Rakuten),
  or the ISIN for the association's CSVs (`JP90C000J569`).
- **Backfill.** Every NAV CSV the bot reads already goes back to the fund's inception, so each
  run rebuilds the NAV and drawdown columns from it; no separate backfill step is needed.
- **Keeping estimates.** `history.json` keeps up to 2,500 scored estimates per fund (about ten
  years), and the estimate error shown on the card still averages only the latest 120.
- **Chart.** The page fetches a fund's history file the first time its card is opened. 3M is
  shown first, in NAV mode. NAV mode draws the published NAV as a line and scored estimates as
  dots; Drawdown mode draws the drawdown line. Dragging across the chart (finger or pointer)
  shows that day's date, NAV, estimate and error, and drawdown. Tapping the card again closes
  it; tapping inside the chart doesn't.

## Acceptance

- HIST-1: Every fund card shows an SVG sparkline of its NAVs over the last 30 days, coloured up
  or down by the move across them.
- HIST-2: Opening the page fetches no history file; opening a card fetches that fund's file once.
- HIST-3: Tapping a fund card opens its chart; tapping the card again closes it. The ★ button
  and controls inside the chart don't toggle it.
- HIST-4: The chart offers 1M, 3M, 1Y and Max, and NAV or Drawdown; 3M and NAV are selected
  first.
- HIST-5: NAV mode shows the published NAVs and the scored estimates; Drawdown mode shows the
  % below the highest NAV up to each day.
- HIST-6: Dragging across the chart shows the date, NAV, estimate, error and drawdown of the
  nearest day.
- HIST-7: Each run writes `history/<fund_id>.json` with every NAV since inception in the format
  above, and drawdown_pct = NAV ÷ highest NAV so far − 1, in %.
- HIST-8: An estimate stays in the history file after it has left the 120 NAVs the card's
  estimate error averages over.
- HIST-9: Each history file is under 50 KB gzipped.
- HIST-10: If a history file can't be fetched, the open card says so; the rest of the page
  works.

## Not doing

- A separate CSV backfill: the NAV CSVs already cover every NAV since inception.
- Estimates before the bot began recording them (early November 2026): they were never made.
- Charts for the markets: their cards already have the forecast PNG.
- A charting library: the chart is an inline SVG drawn by a few lines of script.

## Open

-
