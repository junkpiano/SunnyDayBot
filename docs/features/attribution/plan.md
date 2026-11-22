# Factor attribution — plan

Implements ATTR-1 to ATTR-6 of [spec.md](spec.md).

## Code

```rust
enum Attribution {
    Parts { legs: Vec<(ticker, weight, change)>, fx: Option<change> },  // fx None: hedged
    Etf(ticker, change),  // Tokyo ETF's yen price, can't be split
}
```

| Part | Where |
|---|---|
| Building it | `estimate_funds()`: every part is something the estimate already computed |
| Estimate | `Attribution::ratio()`: `Parts` → (1 + Σ weight × (change − 1)) × fx (1 if hedged); `Etf` → change |
| Line | `Attribution::line()`, with `leg_name()` for the names |
| Card | `fund_card()` puts the line under the change on estimate cards only |

## How it works

- The estimate is NAV × `ratio()`. Because it is computed from the parts, ATTR-5 holds by
  construction, not by a second calculation kept in step.
- **Cases:**

  | Fund | Attribution |
  |---|---|
  | One leg, unhedged | `Parts { legs: [(t, 1, c)], fx: Some(fx) }` |
  | Leveraged, hedged | `Parts { legs: [(t, 3, c)], fx: None }`: `NASDAQ100 -1.50% ×3 · USD/JPY hedged` |
  | Gold Plus (two legs) | `Parts { legs: [(ACWI, 1, c1), (GLDM, 1, c2)], fx }` |
  | Tokyo ETF, traded since the NAV date | `Etf("1540.T", c)` |
  | Tokyo ETF, not traded yet | `Parts { legs: [(first leg, 1, c)], fx: Some(fx) }`, both from hourly bars |
  | NAV already published | None, so no line |
- No new data is fetched. The page grows by about 2 KB (34 lines).

## Tests

- `attributes_estimates`: the line and `ratio()` for one leg unhedged, leveraged hedged, two legs
  and the Tokyo ETF (ATTR-2, 3, 4, 5, 6).
- `shows_the_estimate_and_its_change`, `shows_a_published_nav`: the line on an estimate card and
  none on a published one (ATTR-1).
- One real run on 2026-11-22: eMAXIS Slim 米国株式（S&P500） showed -4.30% and
  `S&P500 -3.70% · USD/JPY -0.63%`; (1 − 0.0370) × (1 − 0.0063) − 1 = −4.31%, the same within
  rounding.
