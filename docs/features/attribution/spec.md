# Factor attribution — spec

## Purpose

Show how much of a fund's estimated change comes from its index and how much from USD/JPY, so
a move like -4.30% reads as "the S&P 500 fell 3.70% and the yen rose 0.63%".

## Behaviour

Under the % change on a card that shows an estimate, one line splits it into its parts:

```
☔ -4.30%
S&P500 -3.70% · USD/JPY -0.63%
```

- **Index part:** each leg's change. A leveraged fund adds its leverage
  (`NASDAQ100 -1.50% ×3`); a fund with two legs shows both (`MSCI ACWI +0.40% · Gold +1.10%`).
- **Currency part:** USD/JPY's change, or `USD/JPY hedged` for a hedged fund.
- **Fund priced from its Tokyo ETF:** once the ETF has traded, its yen change alone
  (`1540 (yen) +0.85%`).
- **Names:** `^GSPC` S&P500, `^NDX` NASDAQ100, `ACWI` MSCI ACWI, `URTH` MSCI World, `^NYFANG`
  FANG+, `^SOX` SOX, `GLDM` and `GC=F` Gold, `1540.T` 1540 (yen); other tickers as they are.
- The parts multiply, not add: (1 + index) × (1 + currency) = 1 + change.

## Acceptance

- ATTR-1: Every card showing an estimate has an attribution line under its % change; a card
  showing a published NAV has none.
- ATTR-2: The index part of a one-leg fund equals the leg's change; a leveraged fund shows the
  leg's change and its leverage.
- ATTR-3: A fund with several legs shows each leg's change.
- ATTR-4: An unhedged fund's currency part is the USD/JPY change the estimate used, including the
  fallback when the NAV date's TTM can't be fetched. A hedged fund shows `USD/JPY hedged`.
- ATTR-5: (1 + index part) × (1 + currency part) − 1 equals the card's % change, before rounding.
- ATTR-6: A fund estimated from its Tokyo ETF's yen price shows that ETF's change alone.

## Not doing

- Splitting a published NAV's change: the fund companies don't publish what moved it.
- Storing the parts in the history files or charting them: only today's estimate is split.
