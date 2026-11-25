# Operations — spec

## Purpose

Keep the site up to date by itself, and never publish worse data than it already has.

## Behaviour

- **Schedule (JST):** 07:17 after the US close, for the day's estimates; 13:17 after MUFG has
  published the day's TTM; every hour from 19:17 to 23:17, as NAVs come out (fund companies in
  the evening, the Investment Trusts Association late at night).
- **Publishing:** each run rebuilds the whole site and publishes it as a static site at
  <https://junkpiano.github.io/SunnyDayBot/>: the page, `history.json`, `history/<fund_id>.json`
  per fund and `<market>_forecast.png` per market.
- **Deploying:** a change on `main` reaches the site at the next scheduled run, once its tests
  have passed.
- **Failures:** a market or fund whose data fails is left out of that run; if the published
  estimate history can't be read, the run stops rather than publish an empty one.

## Contract: data sources

| Data | Source |
|---|---|
| Index, ETF, gold, Bitcoin and USD/JPY prices | Yahoo Finance chart API: `https://query1.finance.yahoo.com/v8/finance/chart/<ticker>?interval=<1d or 60m>&range=<5y, 5d or 1mo>`; closes in `chart.result[0].indicators.quote[0].close`, bar start times in `timestamp` |
| MSCI index levels (MSCI ACWI `892400`, MSCI World `990100`) | MSCI: `https://app2.msci.com/products/service/index/indexmaster/getLevelDataForGraph?currency_symbol=USD&index_variant=NETR&start_date=<yyyymmdd>&end_date=<yyyymmdd>&data_frequency=DAILY&index_codes=<code>`; levels in `indexes.INDEX_LEVELS[]` as `{level_eod, calc_date: yyyymmdd}` |
| Fund NAVs since inception | Each fund's CSV ([funds](../funds/spec.md)): one row per day, first column the date (`2026/11/02` or `2026年11月02日`), second the NAV |
| USD TTM | MUFG: `https://www.murc-kawasesouba.jp/fx/past/index.php?id=<yymmdd>`; the page for that day says `As of <Month> <D>, <YYYY>`, and its USD row has TTS then TTB; a day not archived yet shows another date |

Yahoo is retried twice on HTTP 429, after 2 s and 4 s.

## Contract: estimate history (`history.json`)

Published at the site root and read back by every run, so any implementation must read and write
exactly this format:

```json
{"<fund name>": [
  {"base": "2026-11-02", "runs": [["2026-11-03T22:17:00Z", 37306.2], ...], "estimate": null, "actual": null},
  {"base": "2026-10-30", "estimate": 37100.0, "actual": 37090.0}
]}
```

- `base` is the NAV date the estimate starts from. While the next NAV is pending, every run
  appends `[time, estimate]` to `runs`.
- Once that NAV is out, `runs` is replaced by the chosen `estimate` and the `actual` NAV.
- Up to 2,500 entries are kept per fund; the card's error averages the latest 120.
- Estimates are kept here rather than read back from each published history file: in a trial,
  GitHub Pages answered 503 for 9 of 34 such reads, and a file that can't be read can't be
  rewritten safely.

## Acceptance

- OPS-1: Every change is tested automatically before it can reach the site, and the site is
  generated only from a change whose tests passed.
- OPS-2: A failure to read the live estimate history other than "not found" stops the run before
  anything is written.
- OPS-3: Up to 2,500 estimates are kept per fund.

## Non-functional

- NFR-1: A static site, rebuilt on the schedule above, with no server of its own. Met.

## Not doing

- Describing one particular host's setup in the repository.
