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

- No unit test yet for `fetch_nav()`'s parsing (FUND-2); add one with both date forms if it
  changes.
- FUND-1 and FUND-3 are seen in a generated `site/index.html` and in the run log.
