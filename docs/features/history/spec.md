# History: sparklines and charts — spec

## Purpose

Show where each fund has been, and how close past estimates came, without slowing the first
page load.

## Behaviour

- **Sparkline:** each fund card shows a 30-day NAV sparkline, coloured by the move.
- **Chart:** tapping a card opens a chart under it; tapping the card again closes it.
  - Timeframes 1M, 3M, 1Y and Max; 3M first.
  - **NAV** mode (first): published NAVs as a line, past estimates as dots.
  - **Drawdown** mode: % below the running high.
  - A tooltip follows the finger or pointer: date, NAV, estimate and error, drawdown.
- **History:** goes back to each fund's inception. A fund's history is fetched only when its card
  is first opened, as one file per fund with the fields date, actual_nav, estimated_nav,
  error_jpy, error_pct and drawdown_pct.

## Acceptance

- HIST-1: Every fund card shows an SVG sparkline of its NAVs over the last 30 days, coloured up or
  down by the move across them.
- HIST-2: Opening the page fetches no history file; opening a card fetches that fund's file once.
- HIST-3: Tapping a fund card opens its chart; tapping the card again closes it. The ★ button and
  controls inside the chart don't toggle it.
- HIST-4: The chart offers 1M, 3M, 1Y and Max, and NAV or Drawdown; 3M and NAV are selected first.
- HIST-5: NAV mode shows the published NAVs and the scored estimates; Drawdown mode shows the %
  below the highest NAV up to each day.
- HIST-6: Dragging across the chart shows the date, NAV, estimate, error and drawdown of the
  nearest day.
- HIST-7: Each run writes each fund's history with every NAV since inception, and
  drawdown_pct = NAV ÷ highest NAV so far − 1, in %.
- HIST-8: An estimate stays in the history after it has left the 120 NAVs the card's estimate
  error averages over.
- HIST-9: Each fund's history file is under 50 KB gzipped.
- HIST-10: If a history file can't be fetched, the open card says so; the rest of the page works.

## Non-functional

- NFR-2: Fast first load: only cards and sparklines at first, history on demand. Met:
  `index.html` is 9 KB gzipped and no history is fetched on load.
- NFR-3: History payload under 50 KB gzipped per fund (HIST-9). Met: largest 31 KB
  (三菱UFJ 純金ファンド).

## Not doing

- A separate CSV backfill: the NAV CSVs already cover every NAV since inception.
- History charts for the market cards: they already have the forecast PNG.
- A charting library: the chart is an inline SVG drawn by a few lines of script.
