# Estimation rules

## Purpose

How a fund's next NAV is estimated before it is published. [nav-estimate](nav-estimate.md)
says what the card shows; this file says how the number is made.

## Behaviour

**Which NAV is estimated.** The target is today's date in Japan, or the next weekday on a
weekend. If the fund's latest published NAV is already dated on or after the target, there is
nothing to estimate: the card shows that NAV and its change from the previous one.

**When a NAV is priced.** A Japanese fund holding foreign assets prices its NAV dated D at the
foreign closes of the day before D, converted at D's USD TTM (MUFG's midpoint of TTS and TTB,
set around 10:00 JST).

**Estimate** = latest NAV × index change × currency change, where for a latest NAV dated N:

- **Index change** = 1 + Σ weight × (leg change − 1). A leg's change is its last daily close
  before the target date divided by its last daily close before N. Closes dated on or after
  the target belong to the NAV after it and are ignored.
- **Currency change** (unhedged funds only) = current USD/JPY ÷ N's TTM.
  - Current USD/JPY is the target date's TTM once MUFG has published it; before that, the
    latest close of Yahoo's hourly USD/JPY chart; if that fails, the latest daily close.
    (Yahoo's daily USD/JPY close can be off by most of a yen; the hourly bar isn't.)
  - If N's TTM can't be fetched, the currency change falls back to latest USD/JPY daily close
    ÷ the last daily close before N.
- **Fund holding a Tokyo ETF** (e.g. 三菱UFJ 純金ファンド, which holds 1540): such a fund is
  priced at N's own Tokyo close. Once the ETF has traded after N, the change is the ETF's
  latest close ÷ its close on N. Before that, it is the first leg's change since 15:30 JST on
  N times USD/JPY's change since then, both from hourly bars.

**Scoring an estimate.** Every run records its estimate for the fund's next NAV. A NAV dated D
is priced at closes up to 15:30 JST on D, so once it is published it is scored against the
last estimate made before 15:30 JST on D; later runs already knew moves the NAV doesn't
include. If no run was made before then, that NAV isn't scored. See
[accuracy-and-drawdown](accuracy-and-drawdown.md) for what is shown.

**Weather and colour.** ☀️ and the up colour for a rise, ☔ and the down colour for a fall,
☁️ and grey for no change.

## Acceptance

- EST-1: On Saturday and Sunday the target is Monday; on a weekday it is that day (JST).
- EST-2: An index change uses the last close before the target and the last close before the
  NAV date; a close dated on or after the target is ignored.
- EST-3: A weight of w multiplies the leg's move by w; two legs add their moves.
- EST-4: An unhedged fund's estimate converts from the NAV date's TTM to the target's TTM when
  it is out, otherwise to the latest hourly USD/JPY.
- EST-5: A hedged fund's estimate doesn't move with USD/JPY.
- EST-6: A fund with a Tokyo ETF follows the ETF once it has traded after the NAV date.
- EST-7: A published NAV is scored against the last estimate made before 15:30 JST on its
  date; a run after that is never the one scored.
- EST-8: The TTM parser reads USD TTS and TTB from MUFG's page for the given date, and returns
  nothing when the page is for another date.

## Not doing

Known gaps, each small next to the daily move; correct for them if the error matters:

- Japanese holidays aren't known: on one, the label names that day while the estimate is for
  the next business day.
- Leveraged funds reset daily, so across a holiday the estimate drifts a little.
- Hedge cost is ignored.
- A distribution fund drops by its payout on the record date; the estimate misses that day.
- ETF ex-dividend drops (ACWI, twice a year) aren't corrected for.

## Open

-
