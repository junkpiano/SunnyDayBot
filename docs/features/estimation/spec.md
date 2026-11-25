# Estimation rules — spec

## Purpose

How a fund's next NAV is estimated before it is published, and how each estimate is scored once
the NAV is out.

## Behaviour

- **Which NAV:** the target is today's date in Japan, or the next weekday on a weekend. A fund
  whose latest NAV is already dated on or after the target has nothing to estimate.
- **How a NAV is priced:** a Japanese fund holding foreign assets prices its NAV dated D at the
  foreign closes of the day before D, converted at D's USD TTM (MUFG's midpoint of TTS and TTB,
  set around 10:00 JST).
- **Estimate** = latest NAV × index change × currency change:
  - **Index change** = 1 + Σ weight × (leg change − 1). A leg's change runs from its last close
    before the NAV date to its last close before the target date.
  - **MSCI legs:** a leg standing in for an MSCI index (ACWI for MSCI ACWI, URTH for MSCI World)
    takes its closes from MSCI's published end-of-day levels (net dividends, in USD), which use
    each market's own close, as the fund's index does. The ETF trades only in US hours, so it
    prices a market that was closed (a Japanese holiday, say) a day early. Days MSCI hasn't
    published yet are filled in with the ETF's change since MSCI's latest day; without MSCI's
    levels the leg is the ETF alone.
  - **Currency change** (unhedged funds only) = current USD/JPY ÷ the NAV date's TTM. The
    current rate is the target date's TTM once it is out; before that the latest hourly
    USD/JPY close, or the latest daily close if the hourly one can't be had (daily USD/JPY
    closes can be off by most of a yen, hourly ones aren't). Without the NAV date's TTM, the
    currency change is the latest daily close ÷ the last daily close before the NAV date.
  - **Fund holding a Tokyo ETF:** priced at its NAV date's own Tokyo close. Once the ETF has
    traded since, the estimate is the ETF's yen change since its close on the NAV date. Before
    that, it is the first leg's change times USD/JPY's change, both since 15:30 JST on the NAV
    date, from hourly prices.
- **Scoring:** every run records its estimate. A NAV dated D is priced at closes up to 15:30 JST
  on D, so once it is published it is scored against the last estimate made before then.

## Acceptance

- EST-1: On Saturday and Sunday the target is Monday; on a weekday it is that day (JST).
- EST-2: An index change uses the last close before the target and the last close before the
  NAV date; a close dated on or after the target is ignored.
- EST-3: A weight of w multiplies the leg's move by w; two legs add their moves.
- EST-4: An unhedged fund's estimate converts from the NAV date's TTM to the target's TTM when
  it is out, otherwise to the latest hourly USD/JPY.
- EST-5: A hedged fund's estimate doesn't move with USD/JPY.
- EST-6: A fund with a Tokyo ETF follows the ETF once it has traded after the NAV date.
- EST-7: A published NAV is scored against the last estimate made before 15:30 JST on its date;
  a run after that is never the one scored.
- EST-8: The TTM is read from MUFG's page for the exact date, and nothing is returned when the
  page is for another date.
- EST-9: An MSCI leg follows MSCI's levels up to MSCI's latest published day before the target
  and the ETF's change after it; with no MSCI levels, it follows the ETF.

## Non-functional

- NFR-5: MAPE under 0.25% across the major global and US equity index funds. Met: 0.080% over
  122 NAVs across 10 funds (estimates for the NAVs after 2026-11-02 to 11-19). Gold funds are
  higher, up to about 0.7%.

## Not doing

Known gaps, each small next to the daily move; correct for them if the error matters:

- Japanese holidays: on one, the label names that day while the estimate is for the next
  business day.
- Leveraged funds reset daily, so across a holiday the estimate drifts a little.
- Hedge cost.
- A distribution fund's drop by its payout on the record date.
- ETF ex-dividend drops on a day MSCI hasn't published yet (ACWI, twice a year).
