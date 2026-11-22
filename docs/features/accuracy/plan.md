# Accuracy and drawdown — plan

Implements ACC-1 to ACC-3 of [spec.md](spec.md).

## Code

| Part | Where |
|---|---|
| High | `estimate_funds()`: a fold over the CSV rows that keeps the latest of equal highs |
| Drawdown and badges | `fund_card()` |
| Error | `record()` and `score()`, shown by `accuracy_line()` |

## How it works

- **High:** `rows.fold(rows[0], |h, r| if r.1 >= h.1 { r } else { h })`, so a fund back at its
  high shows today's date.
- **Drawdown:** latest NAV ÷ high − 1, shown with one decimal.
- **Badges:**
  - estimate card: `New high (estimate)` if `estimate.round() > high`, else `High on <NAV date>`
    if the NAV is the high;
  - published card: `High today` if the NAV is the high.
- **Error:** `score()` takes the scored entries (`estimate` and `actual` both set) among the
  latest `SCORE_WINDOW` (120) of the fund's `history.json` entries, and returns their count, mean
  absolute error in 円, mean absolute percentage error and the latest one. With none, the
  footer says "Estimate error: collecting".

## Tests

| Test | Covers |
|---|---|
| `shows_the_high_and_drawdown` | ACC-1, ACC-2 (all three badges) |
| `shows_the_estimate_error` | ACC-3: the line, and "collecting" before any is scored |
| `records_and_scores_estimates` | ACC-3: count, MAE and the latest comparison |
| `keeps_estimates_past_the_score_window` | ACC-3: only the latest 120 are averaged |
