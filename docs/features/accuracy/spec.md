# Accuracy and drawdown — spec

## Purpose

Give each estimate context: where the fund stands against its high, and how close past
estimates came.

## Behaviour

Each fund card's footer shows:

- **High since inception:** the highest NAV, its date and today's drawdown from it
  (`High since inception 39,019円 on 2026-08-14 · -8.8%`).
- **Badges:** `🎉 New high (estimate)` when the estimate passes the high; `🎉 High on <date>` or
  `🎉 High today` when the published NAV is the high.
- **Estimate error:** the average error in 円 and % over the latest 120 scored NAVs, and the
  latest comparison (`Estimate error: avg 70円 (0.19%) over 12 NAVs · after 11/19: 35,558 vs
  actual 35,574`).

## Acceptance

- ACC-1: The card shows the highest NAV since inception, its date (the latest of equal highs) and
  the latest NAV's drawdown from it, in %.
- ACC-2: A card shows `🎉 New high (estimate)` when the estimate, rounded to 円, is above the
  high; `🎉 High on <date>` when the latest published NAV is the high; and `🎉 High today` on a
  published card at the high.
- ACC-3: The card shows the average estimate error in 円 and % over the latest 120 scored NAVs,
  how many were scored, and the latest estimate against its actual NAV; before any is scored,
  it says the error is being collected.

## Not doing

- Estimates from before the bot began recording them (early November 2026): they were never
  made.
