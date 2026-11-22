# Factor Attribution

## Purpose

Show how much of a fund's estimated change comes from its index and how much from USD/JPY, so
a move like -4.09% reads as "the S&P 500 fell 3.2% and the yen rose 0.9%".

## Behaviour

Under the % change on a card that shows an estimate, one line splits it into its parts:

```
☔ -4.09%
S&P500 -3.21% · USD/JPY -0.92%
```

- **Index part:** the index change from [estimation](estimation.md): Σ weight × (leg change − 1).
  - With one leg of weight 1, it shows that leg: `S&P500 -3.21%`.
  - A leveraged fund shows the leg's own move and the leverage: `NASDAQ100 -1.50% ×3`.
  - A fund with several legs shows each one: `MSCI ACWI +0.40% · Gold +1.10%`.
- **Currency part:** USD/JPY's change from the NAV date's TTM to the rate the estimate used,
  for unhedged funds. A hedged fund shows `USD/JPY hedged` instead.
- **Fund priced from its Tokyo ETF** (三菱UFJ 純金ファンド): once the ETF has traded since
  the NAV date, the estimate is the ETF's yen price change, which can't be split. The card
  shows the ETF alone: `1540 (yen) +0.85%`. Before that, it shows the leg and USD/JPY like any
  other fund.
- **How the parts combine:** they multiply, not add: (1 + index) × (1 + currency) = 1 + change.
  The line doesn't show the small cross term.
- **Leg names:** the market names the page already uses (S&P500, NASDAQ, Gold) where there is
  one, otherwise a short name for the ticker: `^NDX` NASDAQ100, `ACWI` MSCI ACWI, `URTH` MSCI
  World, `^NYFANG` FANG+, `^SOX` SOX, `SCHD` SCHD, `GLDM` Gold.

## Acceptance

- ATTR-1: Every card showing an estimate has an attribution line under its % change; a card
  showing a published NAV has none.
- ATTR-2: The index part of a one-leg fund equals the leg's change; a leveraged fund shows the
  leg's change and its leverage.
- ATTR-3: A fund with several legs shows each leg's change.
- ATTR-4: An unhedged fund's currency part is the USD/JPY change the estimate used, including
  the fallback when the NAV date's TTM can't be fetched. A hedged fund shows `USD/JPY hedged`.
- ATTR-5: (1 + index part) × (1 + currency part) − 1 equals the card's % change, before rounding.
- ATTR-6: A fund estimated from its Tokyo ETF's yen price shows that ETF's change alone.

## Not doing

- Attribution of a published NAV's change: the fund companies don't publish what moved it.
- Storing the parts in the history files or charting them: only today's estimate is split.
- Push or webhook notifications: a separate roadmap item.

## Open

-
