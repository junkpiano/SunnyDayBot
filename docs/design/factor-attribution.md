# Design: factor attribution

Implements ATTR-1 to ATTR-6 of the [PRD](../PRD.md#35-factor-attribution-attr).

## Approach

The estimate used to be one number, `ratio`, built up in a loop. Now `estimate_funds()` keeps
what that number was made of, as an `Attribution`, and the estimate is computed from it:

```rust
enum Attribution {
    Parts { legs: Vec<(ticker, weight, change)>, fx: Option<change> },  // fx None: hedged
    Etf(ticker, change),  // Tokyo ETF's yen price, can't be split
}
estimate = NAV × attribution.ratio()
```

- `ratio()` is the formula from [TRD §4.1](../TRD.md#41-estimate-est-16-attr-16):
  - `Parts`: (1 + Σ weight × (change − 1)) × fx, where a hedged fund's fx is 1;
  - `Etf`: the ETF's change.
- Because the estimate is computed from the parts, ATTR-5 (the parts combine to the change)
  holds by construction rather than by a second calculation that has to be kept in step.
- `line()` renders the parts for the card, for example `S&P500 -3.21% · USD/JPY -0.92%`.
  - A leg with a weight other than 1 shows `×weight` after the leg's own change.
  - `leg_name()` maps tickers to the names in the spec; an unknown ticker is shown as is.

No new data is fetched: every part was already computed for the estimate.

## Cases

| Fund | Attribution |
|---|---|
| One leg, unhedged | `Parts { legs: [(t, 1, c)], fx: Some(fx) }` |
| Leveraged, hedged | `Parts { legs: [(t, 3, c)], fx: None }`: `NASDAQ100 -1.50% ×3 · USD/JPY hedged` |
| Gold Plus (two legs) | `Parts { legs: [(ACWI, 1, c1), (GLDM, 1, c2)], fx }` |
| Tokyo ETF, traded since the NAV date | `Etf("1540.T", c)` |
| Tokyo ETF, not traded yet | `Parts { legs: [(first leg, 1, c)], fx: Some(fx) }`, both from hourly bars since 15:30 JST |
| NAV already published | None: no line (ATTR-1) |

The currency part is whatever the estimate used: target TTM ÷ NAV-date TTM, or the live rate
before the target TTM is out, or the daily-close fallback when the NAV-date TTM can't be
fetched (ATTR-4).

## Cost

One short line per card: `index.html` grows by about 2 KB (34 lines of about 60 bytes).

## Testing

- `attributes_estimates` in `src/main.rs` covers the rendered line and `ratio()` for:
  - one leg, unhedged;
  - leveraged, hedged;
  - two legs;
  - the Tokyo ETF.
- One real run on 2026-11-22 showed a line on all 34 cards. For eMAXIS Slim 米国株式（S&P500）,
  the card's change was -4.30% and its line read `S&P500 -3.70% · USD/JPY -0.63%`: (1 − 0.0370)
  × (1 − 0.0063) − 1 = −4.31%, the same within rounding.
