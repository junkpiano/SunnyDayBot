# Chronological History & Time-Series Visualization

- **Card-Level Inline Sparklines:** Render a minimal, non-interactive 30-day SVG sparkline inside each fund card to show recent directional momentum at a glance without cluttering the layout.
- **Expandable Interactive Chart View:** Tapping a fund card expands an interactive time-series chart supporting selectable timeframes (1M, 3M, 1Y, Max).
- **Multi-Series & Drawdown Modes:**
  - **NAV Trend (Actual vs. Estimated):** Plot the official historical NAV alongside historical daily estimates so users can visually inspect tracking accuracy over time.
  - **Drawdown Curve:** Optional toggle displaying historical percentage decline from running All-Time Highs to contextualize market corrections.
- **Daily Automated Archiving & Backfill:**
  - Scheduled CI/CD jobs append daily reconciliation records (date, estimated_nav, actual_nav, error_jpy, error_pct, drawdown_pct) to static JSON files (`data/history/<fund_id>.json`).
  - Support one-time CSV backfilling from official asset manager / Investment Trusts Association (投資信託協会) historical records so multi-year charts work immediately.
