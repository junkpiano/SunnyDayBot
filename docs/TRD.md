# SunnyDay Bot: Technical Requirements Document

How the features in the [PRD](PRD.md) are built and run. Requirement IDs (EST-4, HIST-7, …) are
the PRD's. Design decisions for single features are in [`docs/design/`](design/).

## 1. Architecture

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

## 2. Components

All the code is in `src/main.rs`, and the page template in `src/template.html`.

| Area | Functions | Notes | Implements |
|---|---|---|---|
| Entry | `main` | Markets first, then funds, then writes `index.html` | OPS-2 |
| Market data | `fetch_series`, `fetch_chart` | Yahoo chart API, daily (5y) or hourly bars; retries on 429 | EST-2, MKT-1 |
| Fund NAVs | `fetch_nav` | Parses `2026/11/02` and Shift_JIS `2026年11月02日` rows | FUND-2 |
| USD TTM | `usd_ttm`, `parse_ttm` | MUFG's rate page for a date; None until published | EST-4, EST-8 |
| Estimation | `estimate_funds`, `nav_target`, `change_between`, `change_since`, `change_since_time`, `Attribution` | See §4.1 | NAV-1, EST-1–6, ATTR-1–6, FUND-3 |
| Scoring | `record`, `score` | Keeps every run's estimate and scores the one before 15:30 JST | EST-7, ACC-3, HIST-8, OPS-3 |
| History files | `history_file`, `archive`, `fund_id` | `site/history/<fund_id>.json` | HIST-7, HIST-9 |
| Forecasts | `forecast_series`, `classify`, `render_chart` | Prophet via `augurs` (WASM Stan); charts drawn with `plotters` | MKT-1–3 |
| Page | `build_html`, `fund_card`, `asset_card`, `sparkline`, `accuracy_line`, `Attribution::line` | Fills `src/template.html` | NAV-1–4, ACC-1–3, ATTR-1, HIST-1, FRESH-1 |

Dependencies are `augurs` (Prophet), `plotters` (PNG charts), `reqwest` (blocking HTTP),
`serde_json`, `chrono`, `anyhow` and `urlencoding`.

## 3. Data

### 3.1 Sources

| Data | Source | Used for |
|---|---|---|
| Index, ETF, gold, Bitcoin closes; USD/JPY daily and hourly | Yahoo Finance chart API | Estimates, forecasts |
| Fund NAVs since inception | Fund company CSVs (MUFG, Rakuten) or the Investment Trusts Association library | Latest NAV, high, history, scoring |
| USD TTM | MUFG (`murc-kawasesouba.jp`) | The currency part of estimates |

The fund list, with each fund's CSV, tickers, weights, hedge and Tokyo ETF, is the `FUNDS`
constant; [appendix A](#appendix-a-funds) has it as a table.

### 3.2 `history.json` (state between runs)

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

### 3.3 `history/<fund_id>.json` (per-fund history, served to the chart)

```json
{"fields": ["date", "actual_nav", "estimated_nav", "error_jpy", "error_pct", "drawdown_pct"],
 "rows": [["2026-11-20", 35574.0, 35558.0, -16.0, -0.045, -8.8], ...]}
```

- The file is rebuilt every run from the full NAV CSV and `history.json`, so nothing is
  appended.
- `fund_id` is the CSV's file name (`253425`, `chart_100086`), or the ISIN for the association's
  CSVs.

## 4. Algorithms

### 4.1 Estimate (EST-1–6, ATTR-1–6)

For latest NAV N, dated d, and target date T (today in JST, or the next weekday):

- **Unhedged:** estimate = N × (1 + Σ wᵢ × (legᵢ − 1)) × fx.
- **Hedged:** the same without fx.
- **legᵢ** = last daily close before T ÷ last daily close before d. Closes dated on or after
  T belong to the next NAV.
- **fx** = current rate ÷ d's TTM.
  - The current rate is T's TTM once MUFG has published it. Before that, the latest hourly
    USD/JPY close, or the latest daily close if that fails.
  - If d's TTM can't be fetched, fx = latest daily close ÷ the last daily close before d.
- **Tokyo ETF fund:** once the ETF has traded after d, the estimate is N × the ETF's change
  since its close on d. Before that, it is the first leg × USD/JPY, both from hourly bars since
  15:30 JST on d.

The parts are kept as an `Attribution` (`Parts { legs, fx }` or `Etf`), and the estimate is
computed from it, so the card's attribution line always matches the estimate.

Known gaps, each small next to the daily move; correct for them if the error matters:

- Japanese holidays aren't known: on one, the label names that day while the estimate is for
  the next business day.
- Leveraged funds reset daily, so across a holiday the estimate drifts a little.
- Hedge cost is ignored.
- A distribution fund drops by its payout on the record date; the estimate misses that day.
- ETF ex-dividend drops (ACWI, twice a year) aren't corrected for.

### 4.2 Scoring (EST-7, ACC-3)

A NAV dated D is priced at closes up to 15:30 JST on D, so a published NAV is scored against
the last run before 06:30 UTC on D. A NAV with no run before then isn't scored. MAE and MAPE
are averaged over the latest 120 scored NAVs.

### 4.3 High and drawdown (ACC-1, ACC-2, HIST-7)

- **High:** the highest NAV in the CSV. When the high is tied, the latest date wins, so a fund
  back at its high shows today.
- **Drawdown:** NAV ÷ high − 1. In the history files it is per day, against the running high.

### 4.4 Market forecast (MKT-1, MKT-2)

- **Model:** Prophet, with default settings, trained on the latest 180 daily closes. It
  predicts 30 days ahead.
- **Range:** Prophet's own interval, or ±1.96 standard deviations of the training closes when
  it gives none.
- **Badge:** BUY if the price is below 0.98 × the day-30 forecast, SELL if above 1.02 ×, HOLD
  otherwise.

## 5. Front end

The page is `src/template.html`, filled in with the cards and the update time.

- **Styles:** CSS variables for the light and dark themes (`prefers-color-scheme`), and
  `font-feature-settings: "tnum"` for tabular numerals.
- **Sparklines (HIST-1):** inline SVG drawn by the server; no script needed.
- **Favorites (FAV-1–3):** a localStorage set keyed by card title. Starred cards are copied, without any
  open chart, into the Favorites section.
- **Chart (HIST-2–6, HIST-10):**
  - One delegated click handler covers ★, the chart's controls and the card itself. Cards are
    focusable, and Enter opens them.
  - The first open fetches `history/<fund_id>.json` into a per-page cache. A failed fetch shows
    "History unavailable." and is retried on the next open.
  - The chart is a 320×140 SVG redrawn on each control change: NAV path and estimate dots, or
    the drawdown path.
  - `pointermove` picks the nearest day. `touch-action: pan-y` keeps vertical scrolling.

## 6. Operations (OPS-1–3, NFR-1)

- **Schedule:** a scheduled host runs the bot at 07:17 and 13:17 JST, and hourly from 19:17
  to 23:17.
  - Each run downloads the arm64 binary from the latest successful Build run on `main`. If
    that download fails, it runs the last binary.
  - It then runs the binary and force-pushes `site/` as one commit to `gh-pages`.
- **Why not GitHub Actions:** GitHub drops scheduled runs, and `am.mufg.jp` refuses GitHub's
  runners.
- **Deploying:** pushing to `main` is enough. The next scheduled run uses the new binary once
  CI has passed.
- **Failures:**
  - A market or fund that fails is left out of that run.
  - If USD/JPY fails, no fund is estimated.
  - If `history.json` can't be read (other than 404), the run stops before writing anything, so
    an empty history is never published.
  - If nothing was produced, `index.html` isn't written.

## 7. Build, test and CI

- **Toolchain:** Rust stable; the locked dependencies need rustc 1.95 or newer. Building needs
  `pkg-config` and the fontconfig headers.
- **Tests:** `cargo test --release`. The unit tests cover:
  - target dates;
  - close selection;
  - TTM parsing;
  - recording and scoring;
  - the score window;
  - the history file;
  - fund ids;
  - sparklines;
  - attribution;
  - number formatting.
- **CI:** `.github/workflows/build.yml` runs on `ubuntu-24.04-arm`. It tests every pull request
  and push to `main`, and on `main` keeps the arm64 binary as a run artifact for 90 days.
- **Not in CI:** the page script. It was checked by hand in headless Chromium (Playwright)
  against a locally built site.

## 8. Performance (NFR-2, NFR-3)

| | Measured 2026-11-22 |
|---|---|
| `index.html` | 45 KB (9 KB gzipped) |
| Requests on first load | the page and 5 market PNGs |
| History file per fund | 0.5–31 KB gzipped |
| `history.json` growth | about 600 KB a year until 2,500 entries per fund |

## 9. Security and privacy (NFR-6)

- **No accounts and no tracking.** The browser sends nothing. Favorites stay in localStorage.
- **Read-only inputs.** The bot only reads public sources.
- **Credentials:** the publishing host's GitHub credentials are its only secret, and they
  never enter the repository.
- **Repository contents:** the repository describes the setup, not any one host.

## 10. Design documents

- [History: sparklines and charts](design/history.md)
- [Factor attribution](design/factor-attribution.md)

## Appendix A: Funds

Funds are shown in groups, in the order the groups first appear in the list below. Each fund is
described by:

- **NAV source:** a CSV of every NAV since inception, one `date,NAV` row per day. The fund
  company's own CSV is used where it has one (MUFG and Rakuten publish theirs by the evening);
  otherwise the Investment Trusts Association's library (投資信託協会, `toushin-lib.fwg.ne.jp`,
  by ISIN and association fund code; out late at night).
- **Legs:** the Yahoo Finance tickers it follows, with a weight. The fund's change is the
  weighted sum of the legs' changes: a weight of 2 is a 2x leveraged fund, two legs of 1 are a
  100% + 100% fund. Where an index's own daily closes are unreliable, a liquid ETF stands in
  for it (ACWI, URTH, GLDM).
- **Hedged:** hedged to yen, so USD/JPY doesn't move its NAV.
- **Tokyo ETF:** the Tokyo-listed ETF the fund holds, for a fund priced at its NAV date's own
  Tokyo close (see §4.1).

| Group | Fund | Legs | Hedged | Tokyo ETF | NAV source |
|---|---|---|---|---|---|
| MSCI | eMAXIS Slim 全世界株式（オール・カントリー） | ACWI | no | — | <https://www.am.mufg.jp/fund_file/setteirai/253425.csv> |
| MSCI | 楽天・プラス・オールカントリー株式インデックス・ファンド | ACWI | no | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100086.csv> |
| MSCI | eMAXIS Slim 全世界株式（除く日本） | ACWI | no | — | <https://www.am.mufg.jp/fund_file/setteirai/253209.csv> |
| MSCI | eMAXIS Slim 先進国株式インデックス | URTH | no | — | <https://www.am.mufg.jp/fund_file/setteirai/252653.csv> |
| S&P500 | eMAXIS Slim 米国株式（S&P500） | ^GSPC | no | — | <https://www.am.mufg.jp/fund_file/setteirai/253266.csv> |
| S&P500 | SBI・V・S&P500インデックス・ファンド | ^GSPC | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000J569&associFundCd=89311199> |
| S&P500 | 楽天・プラス・S&P500インデックス・ファンド | ^GSPC | no | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100087.csv> |
| S&P500 | iFreeレバレッジ S&P500 | ^GSPC ×2 | yes | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000GUL6&associFundCd=04315188> |
| NASDAQ100 | ニッセイNASDAQ100インデックスファンド | ^NDX | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000PDY6&associFundCd=29313233> |
| NASDAQ100 | 楽天・プラス・NASDAQ-100インデックス・ファンド | ^NDX | no | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100091.csv> |
| NASDAQ100 | SBI NASDAQ100インデックス・ファンド | ^NDX | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000SWF9&associFundCd=89311265> |
| NASDAQ100 | iFreeレバレッジ NASDAQ100 | ^NDX ×2 | yes | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000H0S5&associFundCd=0431218A> |
| NASDAQ100 | 楽天レバレッジNASDAQ-100 | ^NDX ×2 | yes | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100071.csv> |
| NASDAQ100 | auAMレバレッジNASDAQ100為替ヘッジ無し | ^NDX ×2 | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000QZ77&associFundCd=AY311247> |
| NASDAQ100 | NASDAQ100 3倍ブル | ^NDX ×3 | yes | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000KW84&associFundCd=0431720A> |
| NASDAQ100 | auAMトリプル・NASDAQ100為替ヘッジ無し | ^NDX ×3 | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000T9H5&associFundCd=AY31226A> |
| FANG+ | iFreeNEXT FANG+インデックス | ^NYFANG | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000FZD4&associFundCd=04311181> |
| FANG+ | iFreeレバレッジ FANG+ | ^NYFANG ×2 | yes | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000KLQ6&associFundCd=04314208> |
| SOX | ニッセイSOX指数インデックスファンド（米国半導体株） | ^SOX | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000PDZ3&associFundCd=29314233> |
| SOX | 楽天・プラス・SOXインデックス・ファンド | ^SOX | no | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100092.csv> |
| SCHD | 楽天・シュワブ・高配当株式・米国ファンド（資産成長型） | SCHD | no | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100119.csv> |
| SCHD | 楽天・シュワブ・高配当株式・米国ファンド（四半期決算型） | SCHD | no | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100105.csv> |
| SCHD | SBI・S・米国高配当株式ファンド（年4回決算型） | SCHD | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000REE2&associFundCd=8931224C> |
| Gold | 三菱UFJ 純金ファンド | GC=F | no | 1540.T | <https://www.am.mufg.jp/fund_file/setteirai/251065.csv> |
| Gold | 楽天・ゴールド・ファンド（為替ヘッジなし） | GLDM | no | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100123.csv> |
| Gold | 楽天・ゴールド・ファンド（為替ヘッジあり） | GLDM | yes | — | <https://www.rakuten-toushin.co.jp/assets/csv/chart_100122.csv> |
| Gold | SBI・iシェアーズ・ゴールドファンド（為替ヘッジなし） | GLDM | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000PLM4&associFundCd=8931A236> |
| Gold | SBI・iシェアーズ・ゴールドファンド（為替ヘッジあり） | GLDM | yes | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000PLL6&associFundCd=89319236> |
| Gold | ステート・ストリート・ゴールド・オープン（為替ヘッジなし） | GLDM | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000RXU8&associFundCd=55311258> |
| Gold | アモーヴァ ゴールド・ファンド（為替ヘッジなし） | GLDM | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000F6M7&associFundCd=02312177> |
| Gold | アモーヴァ ゴールド・ファンド（為替ヘッジあり） | GLDM | yes | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000F6L9&associFundCd=02311177> |
| Gold Plus | Tracers MSCIオール・カントリー・ゴールドプラス | ACWI + GLDM | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000SNW3&associFundCd=02311263> |
| Gold Plus | Tracers S&P500ゴールドプラス | ^GSPC + GLDM | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000NS46&associFundCd=02315228> |
| Gold Plus | Tracers NASDAQ100ゴールドプラス | ^NDX + GLDM | no | — | <https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000RH60&associFundCd=02311251> |
