# Operations

## Purpose

How the site is built, published and kept up to date.

## Behaviour

**Build.** One binary run fetches all data, estimates, and writes the site to `site/`:
`index.html` (from `src/template.html`), `history.json` and `<market>_forecast.png`.

**Schedule.** It runs twice a day, at 07:17 JST (after the US close, for the day's estimates)
and 19:17 JST (after most NAVs are out, to show them and score the morning's estimates). The
"published" card points to the 07:17 update.

**Publishing.** A scheduled host runs the arm64 binary built by CI and pushes `site/` as a
single commit to the `gh-pages` branch, which GitHub Pages serves. It doesn't run on GitHub
Actions: GitHub drops scheduled runs, and `am.mufg.jp` refuses GitHub's runners.

**History.** `history.json` holds each fund's recorded and scored estimates (format in
`record()` in `src/main.rs`), the latest 120 per fund. Each run reads it back from the live
site, adds to it, and publishes it again. If the live copy is missing (404) it starts empty;
any other failure stops the run, so an empty history is never published over the real one.

**Data sources.**

| Data | Source |
|---|---|
| Index, ETF, gold, Bitcoin, USD/JPY prices | Yahoo Finance chart API (daily, and hourly where noted) |
| Fund NAVs | Fund company CSVs, or the Investment Trusts Association library ([funds](funds.md)) |
| USD TTM | MUFG's daily rate page (`murc-kawasesouba.jp`) |

**Failures.** Yahoo is retried twice on HTTP 429, after 2 s and 4 s. A market or fund that
fails is left out of that run. If USD/JPY can't be fetched, no fund is estimated. If nothing
at all was produced, `index.html` isn't written.

## Acceptance

- OPS-1: `cargo test --release` passes in CI on every pull request and push to `main`; on
  `main` CI also keeps an arm64 binary as a run artifact.
- OPS-2: A failure to read the live `history.json` other than 404 stops the run before
  anything is written.
- OPS-3: `history.json` keeps at most 120 entries per fund.

## Not doing

- Describing one particular host's setup in the repository.

## Open

-
