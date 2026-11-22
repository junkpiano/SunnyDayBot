# Operations — plan

Implements OPS-1 to OPS-3 of [spec.md](spec.md).

## Code and setup

| Part | Where |
|---|---|
| Run | `main()`: markets, then funds, then `history.json`, the history files and `index.html` under `site/` |
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
- **`load_history()`:** a 404 starts an empty history, and any other error stops the run (OPS-2).
- **Failures inside a run:** a failed market or fund is logged and skipped. If USD/JPY fails, no
  fund is estimated. If nothing was produced, `index.html` isn't written.
- **Yahoo** is retried twice on HTTP 429, after 2 s and 4 s.

## Tests

- OPS-1 is CI itself.
- `drops_old_pending_entries` and `records_and_scores_estimates` cover `record()`.
- OPS-2 and OPS-3's trim have no unit test.
