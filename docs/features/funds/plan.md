# Fund coverage — plan

Implements FUND-1 to FUND-3 of [spec.md](spec.md).

## Code

| Part | Where |
|---|---|
| Fund list | `FUNDS` in `src/main.rs`: `Fund { group, name, url, legs, hedged, tokyo_etf }` per fund, in display order |
| NAV CSV | `fetch_nav()`: every `(date, NAV)` row, oldest first |
| Failures | `estimate_funds()` logs a fund that fails and leaves it out (`Failed to estimate …`) |

## How it works

- `fetch_nav()` reads the CSV as bytes and decodes it lossily, because headers may be Shift_JIS.
  For each line it takes the first column's digit groups as year, month and day, and the second
  column as the NAV. Lines that don't parse are skipped, and an empty result is an error.
- Funds sharing a ticker fetch it once per run.
- Adding a fund: add a `Fund` entry, and its row to the table in the spec.

## Tests

| Test | Covers |
|---|---|
| `parses_nav_csvs` | FUND-2: `parse_nav()` with both date forms and rows to skip |
| `shows_funds_in_their_groups` | FUND-1 (group order) |

FUND-3 (a failing fund is skipped) is in `estimate_funds()`'s network path and is seen in the run
log.
