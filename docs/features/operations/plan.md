# Operations — plan

Implements OPS-1 to OPS-3 of [spec.md](spec.md).

## Architecture

```
            ┌──────────── scheduled host (cron, arm64) ─────────────┐
Yahoo  ───> │                                                        │
MUFG TTM ─> │  sunnydaybot (one Rust binary)  ──>  site/  ──git push─┼─> gh-pages ─> GitHub Pages ─> browser
NAV CSVs ─> │        ▲                                               │                    │
            │        └── history.json read back from the live site ──┼────────────────────┘
            └────────────────────────────────────────────────────────┘
```

- **One process:** a single run fetches every input, computes estimates and forecasts, and
  writes a complete static site. There is no server, database or browser-side API key.
- **State:** the only state carried between runs is `history.json`. Each run reads it back
  from the live site and publishes it again.
- **Browser script:** the page's script is plain JavaScript inlined in `index.html`, with no
  framework and no library. It does favorites and charts.

## Data sources

| Data | Source | Used for |
|---|---|---|
| Index, ETF, gold, Bitcoin closes; USD/JPY daily and hourly | Yahoo Finance chart API | Estimates, forecasts |
| Fund NAVs since inception | Fund company CSVs (MUFG, Rakuten) or the Investment Trusts Association library | Latest NAV, high, history, scoring |
| USD TTM | MUFG (`murc-kawasesouba.jp`) | The currency part of estimates |

The fund list, with each fund's CSV, tickers, weights, hedge and Tokyo ETF, is the `FUNDS`
constant; [funds/spec.md](../funds/spec.md) has it as a table.

## `history.json` (state between runs)

```json
{"<fund name>": [
  {"base": "2026-11-02", "runs": [["2026-11-03T22:17:00Z", 37306.2], ...], "estimate": null, "actual": null},
  {"base": "2026-10-30", "estimate": 37100.0, "actual": 37090.0}
]}
```

- `base` is the NAV date the estimate starts from. While the next NAV is pending, every run
  appends `[time, estimate]` to `runs`.
- Once that NAV is out, `runs` is replaced by the chosen `estimate` and the `actual` NAV.
- Up to 2,500 entries are kept per fund (`HISTORY_KEEP`). The card's error averages the latest
  120 (`SCORE_WINDOW`).
- Estimates are kept here rather than read back from each published history file: in a trial,
  GitHub Pages answered 503 for 9 of 34 such reads, and a file that can't be read can't be
  rewritten safely.

The per-fund history files are described in [history/plan.md](../history/plan.md).

## Security and privacy

- **No accounts and no tracking.** The browser sends nothing. Favorites stay in localStorage.
- **Read-only inputs.** The bot only reads public sources.
- **Credentials:** the publishing host's GitHub credentials are its only secret, and they
  never enter the repository.
- **Repository contents:** the repository describes the setup, not any one host.

## Code and setup

| Part | Where |
|---|---|
| Run | `main()`: reads the published `history.json` first, then markets, funds, the history files and `index.html` under `site/` |
| Estimate history | `load_history()` reads `history.json` from the live site; `record()` trims to `HISTORY_KEEP` |
| CI | `.github/workflows/build.yml` on `ubuntu-24.04-arm`: `cargo test --release` on every pull request and push to `main`; on `main` it also builds the arm64 binary and keeps it as an artifact for 90 days |
| Publishing | A script on the scheduled host, kept outside the repository |

## How it works

- **Each scheduled run** (cron on the host):
  1. downloads the binary from the latest successful Build run on `main`, or keeps the last one
     if that fails;
  2. runs it;
  3. force-pushes `site/` as one commit, "Site YYYY-MM-DD HH:MM", to `gh-pages`, which GitHub
     Pages serves.
- **`load_history()` / `read_history()`:** a 404 starts an empty history, and any other error stops the run before anything is written (OPS-2).
- **Failures inside a run:** a failed market or fund is logged and skipped. If USD/JPY fails, no
  fund is estimated. If nothing was produced, `index.html` isn't written.
- **Yahoo** is retried twice on HTTP 429, after 2 s and 4 s.

## Tests

| Test | Covers |
|---|---|
| CI itself | OPS-1 |
| `reads_the_published_history` | OPS-2: 404 is empty, any other failure or bad body is an error |
| `keeps_at_most_2500_entries` | OPS-3 |
| `records_and_scores_estimates`, `drops_old_pending_entries` | `record()` |

`main()` reads the history before it writes anything, so a failed read leaves `site/` untouched.
