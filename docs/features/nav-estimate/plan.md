# NAV estimate and display — plan

Implements NAV-1 to NAV-4 of [spec.md](spec.md).

## Code

| Part | Where |
|---|---|
| Groups and order | `build_html()` in `src/main.rs`: one section per fund group, in the order the groups first appear in `FUNDS` |
| Card | `fund_card()`: the estimate card, or the published card when `FundResult.estimate` is None |
| Change, icon and colour | `pct_class()` (`up` / `down` / `flat`), `weather()` (☀️ / ☔ / ☁️); colours are CSS variables `--up`, `--down` and `--muted` in `src/template.html` |
| Numbers | `fmt_num()`: thousands separators, no `-0` |
| Target date | `nav_target()`, from the [estimation](../estimation/plan.md) feature |

## How it works

- `estimate_funds()` returns a `FundResult` per fund: latest NAV and date, previous NAV, target
  date, the estimate (None once the target's NAV is published) and what the other features need.
- The estimate card shows `NAV <date>` → `<target> NAV estimate`, and the change is
  estimate ÷ NAV − 1.
- The published card shows the NAV, its change from the previous NAV in 円 and %, and "Published.
  The next estimate comes with the 7:17 update."

## Tests

| Test | Covers |
|---|---|
| `shows_funds_in_their_groups` | NAV-1 (groups in order) |
| `shows_the_estimate_and_its_change` | NAV-1, NAV-2, NAV-3 |
| `shows_a_published_nav` | NAV-4 |
| `formats_numbers` | `fmt_num` |
