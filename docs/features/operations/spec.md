# Operations — spec

## Purpose

Keep the site up to date by itself, and never publish worse data than it already has.

## Behaviour

- **Schedule (JST):** 07:17 after the US close, for the day's estimates; 13:17 after MUFG has
  published the day's TTM; every hour from 19:17 to 23:17, as NAVs come out (fund companies in
  the evening, the Investment Trusts Association late at night).
- **Publishing:** each run rebuilds the whole site and publishes it to GitHub Pages.
- **Deploying:** a change pushed to `main` reaches the site at the next scheduled run, once CI
  has passed.
- **Failures:** a market or fund whose data fails is left out of that run; if the published
  estimate history can't be read, the run stops rather than publish an empty one.

## Acceptance

- OPS-1: Every pull request and push to `main` is tested in CI; on `main` CI also builds the arm64
  binary the site is generated with.
- OPS-2: A failure to read the live estimate history other than "not found" stops the run before
  anything is written.
- OPS-3: Up to 2,500 estimates are kept per fund.

## Non-functional

- NFR-1: A static site on GitHub Pages, rebuilt on the schedule above. Met: built by one binary
  run and published to `gh-pages`.

## Not doing

- Describing one particular host's setup in the repository.
- Running on GitHub Actions' schedule: GitHub drops scheduled runs, and `am.mufg.jp` refuses
  GitHub's runners.
