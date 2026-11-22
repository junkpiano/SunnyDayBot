# Markets

## Purpose

A 30-day outlook for the markets behind the funds, shown below the fund groups.

## Behaviour

Tracked: NASDAQ (`^IXIC`), S&P500 (`^GSPC`), Nikkei225 (`^N225`), Gold (`GC=F`) and Bitcoin
(`BTC-USD`), five years of daily closes from Yahoo Finance.

For each market a Prophet model is trained on the latest 180 daily closes and forecasts the
next 30 days, with an uncertainty range (Prophet's, or ±1.96 standard deviations of the
training closes when Prophet gives none). The card shows:

- the current price and the forecast for day 30, with the % change between them;
- a badge comparing them: **BUY** when the price is more than 2% below the forecast, **SELL**
  when more than 2% above it, otherwise **HOLD**;
- the forecast range for day 30;
- a chart (PNG, 1280×720): five years of closes, the forecast, its range, and today's price
  marked.

## Acceptance

- MKT-1: Each tracked market has a card with price, day-30 forecast, change, badge, range and
  chart.
- MKT-2: The badge is BUY below 98% of the forecast, SELL above 102%, HOLD in between.
- MKT-3: A market whose data or forecast fails is left out; the others still show.

## Not doing

- Advice: the page says the figures are mechanical and not investment advice.

## Open

-
