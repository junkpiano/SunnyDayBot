# NAV estimate and display — spec

## Purpose

Show each fund's likely NAV for today in the morning, hours before the fund company publishes it.

## Behaviour

- Funds are shown in groups: MSCI, S&P500, NASDAQ100, FANG+, SOX, SCHD, Gold, Gold Plus.
- Each card shows the latest official NAV and its date (`NAV 11/20`), an arrow, and the
  estimate of today's NAV (`11/23 NAV estimate`), in 円. On a weekend the estimate is for the
  next weekday.
- Under it, the estimate's % change with a weather icon (☀️ rise, ☔ fall, ☁️ no change),
  coloured green for a rise and red for a fall.
- Once today's NAV is out, the card shows that NAV instead, with its change in 円 and % from the
  previous NAV, and says the next estimate comes with the 07:17 update.

How the estimate is made is the [estimation](../estimation/spec.md) feature.

## Acceptance

- NAV-1: Funds are shown in their groups; each card shows the latest NAV and its date and the
  estimate of the target date's NAV, in 円.
- NAV-2: The estimate's % change is shown with ☀️ for a rise, ☔ for a fall and ☁️ for no
  change.
- NAV-3: A rise is coloured green, a fall red, no change grey.
- NAV-4: Once the target date's NAV is published, the card shows it with its change in 円 and %
  from the previous NAV, and says the next estimate comes with the 07:17 update.

## Not doing

- Investment advice: every figure is a mechanical estimate, and the page says so.
