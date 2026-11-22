# Operations — plan

Implements OPS-1 to OPS-3 of [spec.md](spec.md).

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
