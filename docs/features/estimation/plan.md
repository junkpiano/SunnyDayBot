# Estimation rules — plan

Implements EST-1 to EST-8 of [spec.md](spec.md).

## Code

| Part | Where |
|---|---|
| Target date | `nav_target()` |
| Closes | `fetch_series()` (Yahoo daily, 5 years), `fetch_chart()` (hourly where needed) |
| Changes | `change_between()` (last close before the target ÷ last close before the NAV date), `change_since()`, `change_since_time()` (hourly bars since a time) |
| USD TTM | `usd_ttm()`, `parse_ttm()`: MUFG's page for a date; None until it is published |
| Estimate | `estimate_funds()` builds an `Attribution` and the estimate is NAV × `Attribution::ratio()` |
| Scoring | `record()`, `score()` |

## How it works

For latest NAV N dated d and target date T:

- **Unhedged:** estimate = N × (1 + Σ wᵢ × (legᵢ − 1)) × fx; **hedged:** the same without fx.
- **legᵢ** = `change_between(series, d, T)`: last daily close before T ÷ last daily close before d.
- **fx** = current rate ÷ d's TTM.
  - The current rate is T's TTM once MUFG has published it. Before that, it is the latest hourly
    USD/JPY close, or the latest daily close if the hourly chart fails. Yahoo's daily USD/JPY close can be off by
    most of a yen; the hourly bar isn't.
  - If d's TTM can't be fetched, fx = `change_since(daily USD/JPY, d)`.
  - TTMs are fetched once per date per run (`ttms` map); each index series once per run
    (`cache`).
- **Tokyo ETF fund:** if the ETF has a close after d, the ratio is the ETF's latest close ÷ its
  close on d. Otherwise it is the first leg × USD/JPY, both from hourly bars since 15:30 JST on
  d (`change_since_time`; the base is up to an hour after the close).
- **Scoring:** `record()` appends `[run time, estimate]` under the entry whose `base` is the
  latest NAV date. When the next NAV appears in the CSV, the entry keeps the last run before
  06:30 UTC on that NAV's date as its `estimate`, and the NAV as `actual`; with no such run,
  `estimate` is null and it isn't scored. The format of `history.json` is in the
  [operations plan](../operations/plan.md#historyjson-state-between-runs).

## Tests

| Test | Covers |
|---|---|
| `targets_todays_nav` | EST-1 |
| `ignores_closes_after_the_target` | EST-2 |
| `attributes_estimates` (attribution) | EST-3, EST-5 through `ratio()` |
| `converts_at_the_ttm_once_it_is_out` | EST-4: `now_rate()` and `fx_part()` |
| `follows_the_tokyo_etf_once_it_has_traded` | EST-6: `etf_change()` |
| `records_and_scores_estimates`, `drops_old_pending_entries` | EST-7 |
| `parses_ttm` | EST-8 |

## Limits

The gaps listed under Not doing in the spec. Each is marked in the code with a `ponytail:` comment
where the shortcut is taken.
