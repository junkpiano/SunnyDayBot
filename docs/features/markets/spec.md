# Markets outlook — spec

## Purpose

A 30-day outlook for the markets behind the funds, shown below the fund groups.

## Behaviour

- Markets: NASDAQ, S&P 500, Nikkei 225, gold and Bitcoin.
- Each card shows the current price and a forecast for day 30, with the % change between them,
  the forecast range, a BUY / SELL / HOLD badge, and a chart of five years of prices with the
  forecast.
- **Forecast:** five years of daily closes; a Prophet model (additive trend and seasonality,
  default settings) trained on the latest 180 closes forecasts the next 30 days. The range is
  the model's uncertainty interval, or ±1.96 standard deviations of the training closes when
  it gives none.
- **Chart:** a 1280×720 image of the five years of closes, the forecast and its range, with
  today's price marked.
- The badge is BUY when the price is more than 2% below the forecast, SELL when more than 2%
  above, HOLD otherwise.
- The page says the figures are mechanical and not investment advice.

## Acceptance

- MKT-1: Each tracked market has a card with price, day-30 forecast, change, badge, range and
  chart.
- MKT-2: The badge is BUY below 98% of the forecast, SELL above 102%, HOLD in between.
- MKT-3: A market whose data or forecast fails is left out; the others still show.

## Not doing

- Advice: the figures are mechanical.
