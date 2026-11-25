use anyhow::{Context, Result, anyhow};
use augurs::prophet::wasmstan::WasmstanOptimizer;
use augurs::prophet::{PredictionData, Prophet, TrainingData};
use chrono::{DateTime, Datelike, Duration, NaiveDate, Utc};
use plotters::prelude::*;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Duration as StdDuration;

const HISTORY_YEARS: i64 = 5;
const FORECAST_DAYS: usize = 30;
const TRAIN_DAYS: usize = 180;
const THRESH: f64 = 0.02;
const SITE_DIR: &str = "site";
// Each run reads the published history back from here, adds to it and publishes it again.
const SITE_URL: &str = "https://junkpiano.github.io/SunnyDayBot/";
const HISTORY_FILE: &str = "history.json";
/// Entries kept per fund in history.json, about ten years: they feed the history files.
const HISTORY_KEEP: usize = 2500;
/// The estimate error shown is the average over this many latest NAVs.
const SCORE_WINDOW: usize = 120;
/// Per-fund history files, fetched by the page when a card is opened.
const HISTORY_DIR: &str = "history";
const HISTORY_FIELDS: [&str; 6] = [
    "date",
    "actual_nav",
    "estimated_nav",
    "error_jpy",
    "error_pct",
    "drawdown_pct",
];
const HTTP_RETRIES: usize = 2;
const HTTP_BACKOFF_SECS: u64 = 2;

const TICKERS: &[(&str, &str)] = &[
    ("^IXIC", "NASDAQ"),
    ("^GSPC", "S&P500"),
    ("^N225", "Nikkei225"),
    ("GC=F", "Gold"),
    ("BTC-USD", "Bitcoin"),
];

struct Fund {
    /// Funds with the same group are shown together, in the order they first appear.
    group: &'static str,
    name: &'static str,
    /// NAV history CSV. Every one starts with date,NAV.
    url: &'static str,
    /// What the fund holds, as (Yahoo ticker, weight): its change is the weighted sum of theirs.
    /// A weight of 2 is a 2x leveraged fund; two legs of 1 are a 100% + 100% fund.
    /// ponytail: leveraged funds reset daily, so across a holiday the estimate drifts a little.
    legs: &'static [(&'static str, f64)],
    /// Hedged to yen: USD/JPY doesn't move the NAV. ponytail: the hedge cost is ignored.
    hedged: bool,
    /// The Tokyo-listed ETF the fund holds, for a fund priced at the NAV date's own Tokyo close.
    /// Once the ETF has traded after the NAV date, the estimate follows it; before that, it
    /// measures the first leg and USD/JPY from 15:30 JST on the NAV date.
    tokyo_etf: Option<&'static str>,
}

// NAVs come from the fund company's own CSV where it has one (out by the evening), otherwise from
// the Investment Trusts Association's library (ISIN and association fund code; out late at night).
// ponytail: distribution funds drop by the payout on their record date; the estimate misses that day.
const FUNDS: &[Fund] = &[
    Fund {
        group: "MSCI",
        name: "eMAXIS Slim 全世界株式（オール・カントリー）",
        url: "https://www.am.mufg.jp/fund_file/setteirai/253425.csv",
        legs: &[("ACWI", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "MSCI",
        name: "楽天・プラス・オールカントリー株式インデックス・ファンド",
        url: "https://www.rakuten-toushin.co.jp/assets/csv/chart_100086.csv",
        legs: &[("ACWI", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "MSCI",
        name: "eMAXIS Slim 全世界株式（除く日本）",
        url: "https://www.am.mufg.jp/fund_file/setteirai/253209.csv",
        legs: &[("ACWI", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "MSCI",
        name: "eMAXIS Slim 先進国株式インデックス",
        url: "https://www.am.mufg.jp/fund_file/setteirai/252653.csv",
        legs: &[("URTH", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "S&P500",
        name: "eMAXIS Slim 米国株式（S&P500）",
        url: "https://www.am.mufg.jp/fund_file/setteirai/253266.csv",
        legs: &[("^GSPC", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "S&P500",
        name: "SBI・V・S&P500インデックス・ファンド",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000J569&associFundCd=89311199",
        legs: &[("^GSPC", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "S&P500",
        name: "楽天・プラス・S&P500インデックス・ファンド",
        url: "https://www.rakuten-toushin.co.jp/assets/csv/chart_100087.csv",
        legs: &[("^GSPC", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "S&P500",
        name: "iFreeレバレッジ S&P500",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000GUL6&associFundCd=04315188",
        legs: &[("^GSPC", 2.0)],
        hedged: true,
        tokyo_etf: None,
    },
    Fund {
        group: "NASDAQ100",
        name: "ニッセイNASDAQ100インデックスファンド",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000PDY6&associFundCd=29313233",
        legs: &[("^NDX", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "NASDAQ100",
        name: "楽天・プラス・NASDAQ-100インデックス・ファンド",
        url: "https://www.rakuten-toushin.co.jp/assets/csv/chart_100091.csv",
        legs: &[("^NDX", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "NASDAQ100",
        name: "SBI NASDAQ100インデックス・ファンド",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000SWF9&associFundCd=89311265",
        legs: &[("^NDX", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "NASDAQ100",
        name: "iFreeレバレッジ NASDAQ100",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000H0S5&associFundCd=0431218A",
        legs: &[("^NDX", 2.0)],
        hedged: true,
        tokyo_etf: None,
    },
    Fund {
        group: "NASDAQ100",
        name: "楽天レバレッジNASDAQ-100",
        url: "https://www.rakuten-toushin.co.jp/assets/csv/chart_100071.csv",
        legs: &[("^NDX", 2.0)],
        hedged: true,
        tokyo_etf: None,
    },
    Fund {
        group: "NASDAQ100",
        name: "auAMレバレッジNASDAQ100為替ヘッジ無し",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000QZ77&associFundCd=AY311247",
        legs: &[("^NDX", 2.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "NASDAQ100",
        name: "NASDAQ100 3倍ブル",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000KW84&associFundCd=0431720A",
        legs: &[("^NDX", 3.0)],
        hedged: true,
        tokyo_etf: None,
    },
    Fund {
        group: "NASDAQ100",
        name: "auAMトリプル・NASDAQ100為替ヘッジ無し",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000T9H5&associFundCd=AY31226A",
        legs: &[("^NDX", 3.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "FANG+",
        name: "iFreeNEXT FANG+インデックス",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000FZD4&associFundCd=04311181",
        legs: &[("^NYFANG", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "FANG+",
        name: "iFreeレバレッジ FANG+",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000KLQ6&associFundCd=04314208",
        legs: &[("^NYFANG", 2.0)],
        hedged: true,
        tokyo_etf: None,
    },
    Fund {
        group: "SOX",
        name: "ニッセイSOX指数インデックスファンド（米国半導体株）",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000PDZ3&associFundCd=29314233",
        legs: &[("^SOX", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "SOX",
        name: "楽天・プラス・SOXインデックス・ファンド",
        url: "https://www.rakuten-toushin.co.jp/assets/csv/chart_100092.csv",
        legs: &[("^SOX", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "SCHD",
        name: "楽天・シュワブ・高配当株式・米国ファンド（資産成長型）",
        url: "https://www.rakuten-toushin.co.jp/assets/csv/chart_100119.csv",
        legs: &[("SCHD", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "SCHD",
        name: "楽天・シュワブ・高配当株式・米国ファンド（四半期決算型）",
        url: "https://www.rakuten-toushin.co.jp/assets/csv/chart_100105.csv",
        legs: &[("SCHD", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "SCHD",
        name: "SBI・S・米国高配当株式ファンド（年4回決算型）",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000REE2&associFundCd=8931224C",
        legs: &[("SCHD", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "Gold",
        name: "三菱UFJ 純金ファンド",
        url: "https://www.am.mufg.jp/fund_file/setteirai/251065.csv",
        legs: &[("GC=F", 1.0)],
        hedged: false,
        tokyo_etf: Some("1540.T"),
    },
    Fund {
        group: "Gold",
        name: "楽天・ゴールド・ファンド（為替ヘッジなし）",
        url: "https://www.rakuten-toushin.co.jp/assets/csv/chart_100123.csv",
        legs: &[("GLDM", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "Gold",
        name: "楽天・ゴールド・ファンド（為替ヘッジあり）",
        url: "https://www.rakuten-toushin.co.jp/assets/csv/chart_100122.csv",
        legs: &[("GLDM", 1.0)],
        hedged: true,
        tokyo_etf: None,
    },
    Fund {
        group: "Gold",
        name: "SBI・iシェアーズ・ゴールドファンド（為替ヘッジなし）",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000PLM4&associFundCd=8931A236",
        legs: &[("GLDM", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "Gold",
        name: "SBI・iシェアーズ・ゴールドファンド（為替ヘッジあり）",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000PLL6&associFundCd=89319236",
        legs: &[("GLDM", 1.0)],
        hedged: true,
        tokyo_etf: None,
    },
    Fund {
        group: "Gold",
        name: "ステート・ストリート・ゴールド・オープン（為替ヘッジなし）",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000RXU8&associFundCd=55311258",
        legs: &[("GLDM", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "Gold",
        name: "アモーヴァ ゴールド・ファンド（為替ヘッジなし）",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000F6M7&associFundCd=02312177",
        legs: &[("GLDM", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "Gold",
        name: "アモーヴァ ゴールド・ファンド（為替ヘッジあり）",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000F6L9&associFundCd=02311177",
        legs: &[("GLDM", 1.0)],
        hedged: true,
        tokyo_etf: None,
    },
    Fund {
        group: "Gold Plus",
        name: "Tracers MSCIオール・カントリー・ゴールドプラス",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000SNW3&associFundCd=02311263",
        legs: &[("ACWI", 1.0), ("GLDM", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "Gold Plus",
        name: "Tracers S&P500ゴールドプラス",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000NS46&associFundCd=02315228",
        legs: &[("^GSPC", 1.0), ("GLDM", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
    Fund {
        group: "Gold Plus",
        name: "Tracers NASDAQ100ゴールドプラス",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000RH60&associFundCd=02311251",
        legs: &[("^NDX", 1.0), ("GLDM", 1.0)],
        hedged: false,
        tokyo_etf: None,
    },
];
const USDJPY: &str = "JPY=X";
/// MUFG's daily rate page; a day not archived yet redirects to the current page.
const TTM_URL: &str = "https://www.murc-kawasesouba.jp/fx/past/index.php?id=";

#[derive(Clone, Copy)]
struct SeriesPoint {
    date: NaiveDate,
    /// Start of the bar.
    at: DateTime<Utc>,
    close: f64,
}

#[derive(Clone, Copy)]
struct ForecastPoint {
    date: NaiveDate,
    median: f64,
    upper: f64,
    lower: f64,
}

struct AssetResult {
    name: String,
    actual: f64,
    median: f64,
    upper: f64,
    lower: f64,
    action: String,
}

struct FundResult {
    group: &'static str,
    name: String,
    /// Names the fund's file under `history/`.
    id: String,
    date: NaiveDate,
    nav: f64,
    /// The NAV before `nav`.
    prev: (NaiveDate, f64),
    /// The NAV date estimated: today in Japan, or the next weekday on a weekend.
    target: NaiveDate,
    /// None once the `target` NAV is published: the one after it needs closes that haven't happened.
    estimate: Option<f64>,
    /// What moved `estimate`; None with it.
    attribution: Option<Attribution>,
    high_date: NaiveDate,
    high: f64,
    /// NAVs of the last 30 days, oldest first.
    spark: Vec<f64>,
    accuracy: Accuracy,
}

/// What an estimate is made of: estimate = NAV × `ratio()`.
#[derive(Debug, PartialEq)]
enum Attribution {
    /// (ticker, weight, change) of each leg, and USD/JPY's change; None for a hedged fund.
    Parts {
        legs: Vec<(&'static str, f64, f64)>,
        fx: Option<f64>,
    },
    /// The change of the Tokyo ETF's yen price, which can't be split.
    Etf(&'static str, f64),
}

impl Attribution {
    fn ratio(&self) -> f64 {
        match self {
            Attribution::Parts { legs, fx } => {
                (1.0 + legs.iter().map(|&(_, w, c)| w * (c - 1.0)).sum::<f64>()) * fx.unwrap_or(1.0)
            }
            Attribution::Etf(_, change) => *change,
        }
    }

    /// "S&P500 -3.21% · USD/JPY -0.92%"
    fn line(&self) -> String {
        let pct = |c: f64| format!("{:+.2}%", (c - 1.0) * 100.0);
        match self {
            Attribution::Parts { legs, fx } => legs
                .iter()
                .map(|&(t, w, c)| match w {
                    1.0 => format!("{} {}", leg_name(t), pct(c)),
                    _ => format!("{} {} ×{}", leg_name(t), pct(c), w),
                })
                .chain(std::iter::once(match fx {
                    Some(fx) => format!("USD/JPY {}", pct(*fx)),
                    None => "USD/JPY hedged".to_string(),
                }))
                .collect::<Vec<_>>()
                .join(" · "),
            Attribution::Etf(t, c) => format!("{} {}", leg_name(t), pct(*c)),
        }
    }
}

fn leg_name(ticker: &str) -> &str {
    match ticker {
        "^GSPC" => "S&P500",
        "^NDX" => "NASDAQ100",
        "ACWI" => "MSCI ACWI",
        "URTH" => "MSCI World",
        "^NYFANG" => "FANG+",
        "^SOX" => "SOX",
        "GLDM" | "GC=F" => "Gold",
        "1540.T" => "1540 (yen)",
        other => other,
    }
}

/// How past estimates compared with the NAVs that were then published.
struct Accuracy {
    count: usize,
    mae: f64,
    mape: f64,
    /// (NAV date, estimate, actual) of the latest checked estimate.
    last: Option<(NaiveDate, f64, f64)>,
}

fn main() -> Result<()> {
    // Stop rather than publish an empty history over the real one, before writing anything.
    let mut history = load_history().context("load the published history")?;

    let site_dir = Path::new(SITE_DIR);
    fs::create_dir_all(site_dir).context("create site directory")?;

    let mut results: Vec<AssetResult> = Vec::new();

    for (ticker, name) in TICKERS {
        println!("Processing {}...", name);
        let series = match fetch_series(ticker) {
            Ok(series) => series,
            Err(err) => {
                eprintln!("Failed to fetch {}: {:#}", name, err);
                continue;
            }
        };
        if series.is_empty() {
            println!("No valid data for {}. Skipping.", name);
            continue;
        }

        let (forecast, actual_price) = match forecast_series(&series) {
            Ok(fit) => fit,
            Err(err) => {
                eprintln!("Failed to forecast {}: {:#}", name, err);
                continue;
            }
        };
        let last_forecast = forecast
            .last()
            .ok_or_else(|| anyhow!("forecast empty for {}", name))?;

        let action = classify(actual_price, last_forecast.median);

        results.push(AssetResult {
            name: name.to_string(),
            actual: round2(actual_price),
            median: round2(last_forecast.median),
            upper: round2(last_forecast.upper),
            lower: round2(last_forecast.lower),
            action: action.to_string(),
        });

        render_chart(name, &series, &forecast)?;
    }

    let funds = estimate_funds(&mut history);
    fs::write(
        site_dir.join(HISTORY_FILE),
        serde_json::to_string(&history).context("serialize history")?,
    )
    .context("write history")?;

    if results.is_empty() && funds.is_empty() {
        println!("No forecasts were produced (all data unavailable).");
        return Ok(());
    }

    fs::write(site_dir.join("index.html"), build_html(&results, &funds))
        .context("write site/index.html")?;
    println!("Wrote {}/index.html", SITE_DIR);

    Ok(())
}

fn fetch_series(ticker: &str) -> Result<Vec<SeriesPoint>> {
    fetch_chart(ticker, "1d", "5y")
}

fn fetch_chart(ticker: &str, interval: &str, range: &str) -> Result<Vec<SeriesPoint>> {
    let now = Utc::now().date_naive();
    let start = now - Duration::days(365 * HISTORY_YEARS);

    let encoded = urlencoding::encode(ticker);
    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{}?interval={}&range={}",
        encoded, interval, range
    );

    let client = reqwest::blocking::Client::builder()
        .user_agent("Mozilla/5.0 (compatible; SunnyDayBot/1.0)")
        .build()
        .context("build http client")?;

    let mut last_error: Option<anyhow::Error> = None;
    for attempt in 0..=HTTP_RETRIES {
        let response = client
            .get(&url)
            .send()
            .with_context(|| format!("request yahoo finance for {}", ticker))?;
        let status = response.status();
        let text = response
            .text()
            .unwrap_or_else(|_| "<failed to read body>".to_string());

        if !status.is_success() {
            let err = anyhow!("yahoo finance returned {} for {}: {}", status, ticker, text);
            if status.as_u16() == 429 && attempt < HTTP_RETRIES {
                let backoff = HTTP_BACKOFF_SECS * (attempt as u64 + 1);
                eprintln!(
                    "Rate limited for {} (attempt {}/{}). Retrying in {}s...",
                    ticker,
                    attempt + 1,
                    HTTP_RETRIES + 1,
                    backoff
                );
                std::thread::sleep(StdDuration::from_secs(backoff));
                last_error = Some(err);
                continue;
            }
            return Err(err);
        }

        let resp: Value = serde_json::from_str(&text).context("parse yahoo finance json")?;

        let result = resp
            .get("chart")
            .and_then(|v| v.get("result"))
            .and_then(|v| v.get(0))
            .ok_or_else(|| anyhow!("missing chart result for {}", ticker))?;

        let timestamps = result
            .get("timestamp")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing timestamps for {}", ticker))?;

        let closes = result
            .get("indicators")
            .and_then(|v| v.get("quote"))
            .and_then(|v| v.get(0))
            .and_then(|v| v.get("close"))
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("missing close data for {}", ticker))?;

        let mut series = Vec::new();
        for (ts_value, close_value) in timestamps.iter().zip(closes.iter()) {
            let ts = match ts_value.as_i64() {
                Some(v) => v,
                None => continue,
            };
            let close = match close_value.as_f64() {
                Some(v) if v.is_finite() => v,
                _ => continue,
            };

            let dt = match DateTime::<Utc>::from_timestamp(ts, 0) {
                Some(v) => v,
                None => continue,
            };
            let date = dt.date_naive();
            if date < start {
                continue;
            }

            series.push(SeriesPoint {
                date,
                at: dt,
                close,
            });
        }

        return Ok(series);
    }

    Err(last_error.unwrap_or_else(|| anyhow!("request failed for {}", ticker)))
}

/// Each fund's estimate of today's NAV; funds whose data is unavailable are left out.
fn estimate_funds(history: &mut Value) -> Vec<FundResult> {
    let target = nav_target(Utc::now());
    let fx = match fetch_series(USDJPY) {
        Ok(fx) => fx,
        Err(err) => {
            eprintln!("Failed to fetch USD/JPY: {:#}", err);
            return Vec::new();
        }
    };

    // Today's TTM once it is out, the live rate before that.
    // Yahoo's daily USD/JPY closes can be off by most of a yen; the latest hourly bar isn't.
    let live = || {
        fetch_chart(USDJPY, "60m", "5d")
            .ok()
            .and_then(|s| s.last().map(|p| p.close))
            .or(fx.last().map(|p| p.close))
            .unwrap_or(f64::NAN)
    };
    let fx_now = now_rate(
        usd_ttm(target).inspect_err(|e| eprintln!("Failed to fetch today's TTM: {:#}", e)),
        live,
    );
    let mut ttms: HashMap<NaiveDate, Option<f64>> = HashMap::new();

    // Many funds share an index; fetch each once.
    let mut cache: HashMap<&str, Vec<SeriesPoint>> = HashMap::new();
    // MSCI's levels per index; None when they couldn't be fetched, so the ETF alone is used.
    let mut msci_cache: HashMap<&str, Option<Vec<SeriesPoint>>> = HashMap::new();
    let mut funds = Vec::new();
    for f in FUNDS {
        let name = f.name;
        type Fetched = (
            Option<Attribution>,
            Vec<(NaiveDate, f64)>,
            (NaiveDate, f64),
            NaiveDate,
            f64,
        );
        let fund = (|| -> Result<Fetched> {
            let rows = fetch_nav(f.url)?;
            let date = rows.last().unwrap().0;
            let prev = rows[rows.len().saturating_sub(2)];
            // The latest of equal highs, so a fund back at its high shows today's date.
            let (high_date, high) = rows
                .iter()
                .copied()
                .fold(rows[0], |h, r| if r.1 >= h.1 { r } else { h });
            if date >= target {
                return Ok((None, rows, prev, high_date, high));
            }
            let attribution = if let Some(etf) = f.tokyo_etf {
                if let Some(change) = etf_change(&fetch_series(etf)?, date)? {
                    Attribution::Etf(etf, change)
                } else {
                    // Tokyo hasn't traded since. 15:30 JST. ponytail: hourly bars, so the base
                    // is up to an hour after the close.
                    let close = date.and_hms_opt(6, 30, 0).unwrap().and_utc();
                    let leg = change_since_time(&fetch_chart(f.legs[0].0, "60m", "1mo")?, close)?;
                    let fx = change_since_time(&fetch_chart(USDJPY, "60m", "1mo")?, close)?;
                    Attribution::Parts {
                        legs: vec![(f.legs[0].0, 1.0, leg)],
                        fx: Some(fx),
                    }
                }
            } else {
                // Closes before the target date: later ones belong to the NAV after it.
                let mut legs = Vec::new();
                for &(ticker, weight) in f.legs {
                    let series = match cache.get(ticker) {
                        Some(series) => series,
                        None => cache.entry(ticker).or_insert(fetch_series(ticker)?),
                    };
                    let msci = msci_code(ticker).and_then(|code| {
                        msci_cache
                            .entry(code)
                            .or_insert_with(|| {
                                fetch_msci(code)
                                    .inspect_err(|e| {
                                        eprintln!("Failed to fetch MSCI {}: {:#}", code, e)
                                    })
                                    .ok()
                            })
                            .as_deref()
                    });
                    let change = match msci {
                        Some(msci) => msci_change(msci, series, date, target)?,
                        None => change_between(series, date, target)?,
                    };
                    legs.push((ticker, weight, change));
                }
                let fx = if f.hedged {
                    None
                } else {
                    // The NAV was converted at that day's TTM.
                    let base = *ttms.entry(date).or_insert_with(|| {
                        usd_ttm(date)
                            .inspect_err(|e| eprintln!("Failed to fetch TTM for {}: {:#}", date, e))
                            .ok()
                            .flatten()
                    });
                    Some(fx_part(base, fx_now, &fx, date)?)
                };
                Attribution::Parts { legs, fx }
            };
            Ok((Some(attribution), rows, prev, high_date, high))
        })()
        .map(|(attribution, rows, prev, high_date, high)| {
            let estimate = attribution
                .as_ref()
                .map(|a| rows.last().unwrap().1 * a.ratio());
            let accuracy = record(history, name, &rows, estimate, Utc::now());
            let id = fund_id(f.url);
            if let Err(e) = archive(&id, &rows, &history[name]) {
                eprintln!("Failed to write the history of {}: {:#}", name, e);
            }
            let (date, nav) = *rows.last().unwrap();
            let month_ago = date - Duration::days(30);
            FundResult {
                group: f.group,
                name: name.to_string(),
                id,
                date,
                nav,
                prev,
                target,
                estimate,
                attribution,
                high_date,
                high,
                spark: rows
                    .iter()
                    .filter(|r| r.0 >= month_ago)
                    .map(|r| r.1)
                    .collect(),
                accuracy,
            }
        });
        match fund {
            Ok(fund) => funds.push(fund),
            Err(err) => eprintln!("Failed to estimate {}: {:#}", name, err),
        }
    }
    funds
}

/// The published history, or an empty one if none has been published yet.
fn load_history() -> Result<Value> {
    let url = format!("{}{}", SITE_URL, HISTORY_FILE);
    let resp = reqwest::blocking::get(&url).with_context(|| format!("fetch {}", url))?;
    let status = resp.status().as_u16();
    read_history(status, &resp.text()?).with_context(|| format!("read {}", url))
}

/// The published history from its HTTP status and body: empty if it isn't there yet, an error
/// for any other failure, so a run never starts from an empty history by mistake.
fn read_history(status: u16, body: &str) -> Result<Value> {
    match status {
        404 => Ok(json!({})),
        200..=299 => Ok(serde_json::from_str(body)?),
        _ => Err(anyhow!("HTTP {}", status)),
    }
}

/// "253425" or "chart_100086" from a fund company's CSV URL, the ISIN from the association's.
fn fund_id(url: &str) -> String {
    match url.split_once("isinCd=") {
        Some((_, rest)) => rest.split('&').next().unwrap().to_string(),
        None => url
            .rsplit('/')
            .next()
            .unwrap()
            .trim_end_matches(".csv")
            .to_string(),
    }
}

/// Writes `history/<id>.json` (see `history_file`) from the NAV CSV and history.json.
fn archive(id: &str, rows: &[(NaiveDate, f64)], scored: &Value) -> Result<()> {
    let dir = Path::new(SITE_DIR).join(HISTORY_DIR);
    fs::create_dir_all(&dir).context("create history directory")?;
    let scored = scored.as_array().map_or(&[][..], |v| v);
    fs::write(
        dir.join(format!("{}.json", id)),
        serde_json::to_string(&history_file(rows, scored))?,
    )
    .with_context(|| format!("write history of {}", id))
}

/// {"fields": [...HISTORY_FIELDS], "rows": [["2026-11-20", 35574.0, 35558.0, -16.0, -0.045, -8.8], ...]}:
/// every NAV in `rows`, oldest first, with the estimate scored against it (null where there is
/// none) and its error, and the drawdown from the highest NAV up to that day.
fn history_file(rows: &[(NaiveDate, f64)], scored: &[Value]) -> Value {
    let mut estimates: HashMap<String, f64> = HashMap::new();
    for e in scored {
        let base = e["base"].as_str().and_then(|b| b.parse::<NaiveDate>().ok());
        let (Some(base), Some(estimate)) = (base, e["estimate"].as_f64()) else {
            continue;
        };
        if let Some(r) = rows.iter().find(|r| r.0 > base) {
            estimates.insert(r.0.to_string(), estimate);
        }
    }
    let round = |v: f64, d: i32| (v * 10f64.powi(d)).round() / 10f64.powi(d);
    let mut high = f64::MIN;
    let rows: Vec<Value> = rows
        .iter()
        .map(|&(date, nav)| {
            high = high.max(nav);
            let date = date.to_string();
            let drawdown = round((nav / high - 1.0) * 100.0, 2);
            match estimates.get(&date) {
                Some(&est) => json!([
                    date,
                    nav,
                    round(est, 2),
                    round(est - nav, 2),
                    round((est / nav - 1.0) * 100.0, 3),
                    drawdown
                ]),
                None => json!([date, nav, null, null, null, drawdown]),
            }
        })
        .collect();
    json!({"fields": HISTORY_FIELDS, "rows": rows})
}

/// Records this run's `estimate` of the NAV after the latest one in `rows`, and scores past
/// estimates whose NAV is now published. A NAV dated D is priced at closes up to 15:30 JST on D,
/// so a run after that no longer estimates it: the score uses the last run before D 15:30 JST.
/// D is only known once the NAV is out (holidays), so each run's estimate is kept until then.
///
/// history = {"<fund>": [{"base": "2026-11-02", "runs": [["2026-11-03T22:17:00Z", 37306.2], ...],
///                        "estimate": null, "actual": null}, ...]}
/// Once the NAV is out, "runs" is replaced by the chosen "estimate" and the "actual" NAV.
fn record(
    history: &mut Value,
    fund: &str,
    rows: &[(NaiveDate, f64)],
    estimate: Option<f64>,
    now: DateTime<Utc>,
) -> Accuracy {
    let base = rows.last().unwrap().0;
    let map = history.as_object_mut().expect("history is an object");
    let list = map
        .entry(fund)
        .or_insert_with(|| json!([]))
        .as_array_mut()
        .expect("history entries are arrays");
    // Entries without "runs" still pending were recorded before this format and may hold an
    // estimate made after the NAV was priced; drop them.
    list.retain(|e| !e["actual"].is_null() || e["runs"].is_array());

    for e in list.iter_mut().filter(|e| e["actual"].is_null()) {
        let Some(b) = e["base"].as_str().and_then(|b| b.parse::<NaiveDate>().ok()) else {
            continue;
        };
        let Some(&(date, nav)) = rows.iter().find(|r| r.0 > b) else {
            continue;
        };
        let cutoff = date.and_hms_opt(6, 30, 0).unwrap().and_utc().to_rfc3339();
        let runs = e["runs"].as_array().cloned().unwrap_or_default();
        let before: Vec<&Value> = runs
            .iter()
            .filter(|r| r[0].as_str() < Some(&cutoff))
            .collect();
        // No run before the cutoff (the NAV came out between runs): nothing fair to score.
        e["estimate"] = before.last().map_or(Value::Null, |r| r[1].clone());
        e["actual"] = json!(nav);
        e.as_object_mut().unwrap().remove("runs");
    }

    if let Some(estimate) = estimate {
        let base_str = base.to_string();
        let run = json!([now.to_rfc3339(), estimate]);
        match list.iter_mut().find(|e| e["base"] == base_str.as_str()) {
            Some(e) => e["runs"].as_array_mut().unwrap().push(run),
            None => list
                .push(json!({"base": base_str, "runs": [run], "estimate": null, "actual": null})),
        }
    }
    let excess = list.len().saturating_sub(HISTORY_KEEP);
    list.drain(..excess);

    score(&list[list.len().saturating_sub(SCORE_WINDOW)..])
}

fn score(list: &[Value]) -> Accuracy {
    let checked: Vec<(NaiveDate, f64, f64)> = list
        .iter()
        .filter_map(|e| {
            let base = e["base"].as_str()?.parse::<NaiveDate>().ok()?;
            Some((base, e["estimate"].as_f64()?, e["actual"].as_f64()?))
        })
        .collect();
    let n = checked.len().max(1) as f64;
    Accuracy {
        count: checked.len(),
        mae: checked.iter().map(|c| (c.1 - c.2).abs()).sum::<f64>() / n,
        mape: checked
            .iter()
            .map(|c| ((c.1 - c.2) / c.2).abs())
            .sum::<f64>()
            / n
            * 100.0,
        last: checked.last().copied(),
    }
}

/// The MSCI index an ETF leg stands in for: MSCI ACWI and MSCI World.
fn msci_code(ticker: &str) -> Option<&'static str> {
    match ticker {
        "ACWI" => Some("892400"),
        "URTH" => Some("990100"),
        _ => None,
    }
}

/// MSCI's published end-of-day levels (net dividends, USD) for the last 30 days, oldest first.
fn fetch_msci(code: &str) -> Result<Vec<SeriesPoint>> {
    let end = Utc::now().date_naive();
    let url = format!(
        "https://app2.msci.com/products/service/index/indexmaster/getLevelDataForGraph?currency_symbol=USD&index_variant=NETR&start_date={}&end_date={}&data_frequency=DAILY&index_codes={}",
        (end - Duration::days(30)).format("%Y%m%d"),
        end.format("%Y%m%d"),
        code
    );
    let text = reqwest::blocking::Client::builder()
        .user_agent("Mozilla/5.0 (compatible; SunnyDayBot/1.0)")
        .build()?
        .get(&url)
        .send()
        .and_then(|r| r.error_for_status())
        .and_then(|r| r.text())
        .with_context(|| format!("fetch MSCI {}", code))?;
    parse_msci(&text).with_context(|| format!("parse MSCI {}", code))
}

fn parse_msci(text: &str) -> Result<Vec<SeriesPoint>> {
    let resp: Value = serde_json::from_str(text)?;
    let levels = resp["indexes"]["INDEX_LEVELS"]
        .as_array()
        .ok_or_else(|| anyhow!("no INDEX_LEVELS"))?;
    let mut series = Vec::new();
    for l in levels {
        let (Some(close), Some(day)) = (l["level_eod"].as_f64(), l["calc_date"].as_i64()) else {
            continue;
        };
        let date = NaiveDate::parse_from_str(&day.to_string(), "%Y%m%d")?;
        let at = date.and_hms_opt(0, 0, 0).unwrap().and_utc();
        series.push(SeriesPoint { date, at, close });
    }
    if series.is_empty() {
        return Err(anyhow!("no levels"));
    }
    Ok(series)
}

/// An MSCI leg: MSCI's change up to its latest published day before `target`, times the ETF's
/// change after that day.
fn msci_change(
    msci: &[SeriesPoint],
    etf: &[SeriesPoint],
    nav_date: NaiveDate,
    target: NaiveDate,
) -> Result<f64> {
    let last = msci
        .iter()
        .rev()
        .find(|p| p.date < target)
        .ok_or_else(|| anyhow!("no MSCI level before {}", target))?;
    let after = change_between(etf, last.date.succ_opt().unwrap(), target)?;
    Ok(change_between(msci, nav_date, target)? * after)
}

/// The USD TTM (midpoint of TTS and TTB) MUFG published for `date`; None until it is out.
/// Funds convert their NAV at this rate, set around 10:00 JST.
fn usd_ttm(date: NaiveDate) -> Result<Option<f64>> {
    let url = format!("{}{}", TTM_URL, date.format("%y%m%d"));
    let bytes = reqwest::blocking::get(&url)
        .and_then(|r| r.error_for_status())
        .and_then(|r| r.bytes())
        .with_context(|| format!("fetch TTM {}", url))?;
    Ok(parse_ttm(&String::from_utf8_lossy(&bytes), date))
}

fn parse_ttm(html: &str, date: NaiveDate) -> Option<f64> {
    if !html.contains(&date.format("As of %B %-d, %Y").to_string()) {
        return None;
    }
    let row = &html[html.find(">USD</td>")?..];
    let mut nums = row
        .split(|c: char| c == '>' || c == '<')
        .filter_map(|t| t.trim().parse::<f64>().ok());
    Some((nums.next()? + nums.next()?) / 2.0)
}

/// Every (date, NAV) row of a fund's CSV, oldest first. Headers may be Shift_JIS; the data rows
/// are ASCII.
fn fetch_nav(url: &str) -> Result<Vec<(NaiveDate, f64)>> {
    let bytes = reqwest::blocking::get(url)
        .and_then(|r| r.error_for_status())
        .and_then(|r| r.bytes())
        .with_context(|| format!("fetch NAV csv {}", url))?;
    let rows = parse_nav(&String::from_utf8_lossy(&bytes));
    if rows.is_empty() {
        return Err(anyhow!("no NAV rows in csv {}", url));
    }
    Ok(rows)
}

/// The (date, NAV) rows of a NAV CSV; lines that aren't a date and a number are skipped.
fn parse_nav(text: &str) -> Vec<(NaiveDate, f64)> {
    text.lines()
        .filter_map(|line| {
            let mut cols = line.split(',');
            // "2026/11/02" or "2026年11月02日" (Shift_JIS, so the kanji are mangled here).
            let mut ymd = cols
                .next()?
                .split(|c: char| !c.is_ascii_digit())
                .filter(|p| !p.is_empty());
            let date = NaiveDate::from_ymd_opt(
                ymd.next()?.parse().ok()?,
                ymd.next()?.parse().ok()?,
                ymd.next()?.parse().ok()?,
            )?;
            let nav = cols.next()?.parse::<f64>().ok()?;
            Some((date, nav))
        })
        .collect()
}

/// Latest close / last close dated before `nav_date`. A NAV for Japan date D is priced at the
/// foreign closes of the day before D and at D's morning USD/JPY, which is the close before D.
// ponytail: ignores ETF ex-dividend drops on days MSCI hasn't published (ACWI, twice a year); correct for them if the error matters.
/// Latest close / close of the last bar that started before `at`.
/// The rate an estimate converts to: the target date's TTM once it is out, else `live()`.
fn now_rate(target_ttm: Result<Option<f64>>, live: impl FnOnce() -> f64) -> f64 {
    match target_ttm {
        Ok(Some(ttm)) => ttm,
        _ => live(),
    }
}

/// USD/JPY's change since the NAV was converted: from the NAV date's TTM to `now`, or along the
/// daily closes when that TTM couldn't be fetched.
fn fx_part(
    nav_ttm: Option<f64>,
    now: f64,
    daily: &[SeriesPoint],
    nav_date: NaiveDate,
) -> Result<f64> {
    match nav_ttm {
        Some(base) => Ok(now / base),
        None => change_since(daily, nav_date),
    }
}

/// A Tokyo ETF's change since its close on `nav_date`; None until it has traded after it.
fn etf_change(series: &[SeriesPoint], nav_date: NaiveDate) -> Result<Option<f64>> {
    if !series.last().is_some_and(|p| p.date > nav_date) {
        return Ok(None);
    }
    change_since(series, nav_date.succ_opt().unwrap()).map(Some)
}

fn change_since_time(series: &[SeriesPoint], at: DateTime<Utc>) -> Result<f64> {
    let base = series
        .iter()
        .rev()
        .find(|p| p.at < at)
        .ok_or_else(|| anyhow!("no bar before {}", at))?;
    let latest = series.last().ok_or_else(|| anyhow!("empty series"))?;
    Ok(latest.close / base.close)
}

/// Today's date in Japan, or the next weekday on a weekend. ponytail: Japanese holidays aren't
/// known, so on one the label says that day while the estimate is for the next business day.
fn nav_target(now: DateTime<Utc>) -> NaiveDate {
    let mut d = (now + Duration::hours(9)).date_naive();
    while d.weekday().number_from_monday() > 5 {
        d = d.succ_opt().unwrap();
    }
    d
}

/// Change from the last close before `nav_date` to the last close before `target`.
fn change_between(series: &[SeriesPoint], nav_date: NaiveDate, target: NaiveDate) -> Result<f64> {
    let before = |d: NaiveDate| {
        series
            .iter()
            .rev()
            .find(|p| p.date < d)
            .ok_or_else(|| anyhow!("no close before {}", d))
    };
    Ok(before(target)?.close / before(nav_date)?.close)
}

fn change_since(series: &[SeriesPoint], nav_date: NaiveDate) -> Result<f64> {
    let base = series
        .iter()
        .rev()
        .find(|p| p.date < nav_date)
        .ok_or_else(|| anyhow!("no close before {}", nav_date))?;
    let latest = series.last().ok_or_else(|| anyhow!("empty series"))?;
    Ok(latest.close / base.close)
}

fn forecast_series(series: &[SeriesPoint]) -> Result<(Vec<ForecastPoint>, f64)> {
    let len = series.len();
    if len < 2 {
        return Err(anyhow!("not enough data to forecast"));
    }

    let actual_price = series[len - 1].close;

    let start = if len > TRAIN_DAYS {
        len - TRAIN_DAYS
    } else {
        0
    };
    let train = &series[start..];

    let mut ds: Vec<i64> = Vec::with_capacity(train.len());
    let mut y: Vec<f64> = Vec::with_capacity(train.len());
    for p in train {
        ds.push(date_to_ts(p.date)?);
        y.push(p.close);
    }

    let data = TrainingData::new(ds, y.clone()).context("create prophet training data")?;
    let mut prophet = Prophet::new(Default::default(), WasmstanOptimizer::new());
    prophet
        .fit(data, Default::default())
        .context("fit prophet model")?;

    let last_date = series[len - 1].date;
    let mut future_ts: Vec<i64> = Vec::with_capacity(FORECAST_DAYS);
    for i in 1..=FORECAST_DAYS {
        let date = last_date + Duration::days(i as i64);
        future_ts.push(date_to_ts(date)?);
    }

    let prediction_data = PredictionData::new(future_ts);
    let predictions = prophet
        .predict(Some(prediction_data))
        .context("predict with prophet")?;

    let yhat = predictions.yhat;
    if yhat.point.is_empty() {
        return Err(anyhow!("forecast returned no points"));
    }

    let points = yhat.point;
    let mut lower = yhat.lower.unwrap_or_default();
    let mut upper = yhat.upper.unwrap_or_default();
    if lower.len() != points.len() || upper.len() != points.len() {
        let interval = 1.96 * stddev(&y);
        lower = points.iter().map(|v| v - interval).collect();
        upper = points.iter().map(|v| v + interval).collect();
    }

    let steps = points.len().min(FORECAST_DAYS);
    if steps == 0 {
        return Err(anyhow!("forecast returned no usable points"));
    }

    let mut out = Vec::with_capacity(steps);
    for i in 0..steps {
        let date = last_date + Duration::days((i + 1) as i64);
        let median = points[i];
        let lower = lower[i];
        let upper = upper[i];
        out.push(ForecastPoint {
            date,
            median,
            upper,
            lower,
        });
    }

    Ok((out, actual_price))
}

fn stddev(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / n as f64;
    let mut sum_sq = 0.0;
    for v in values {
        let diff = v - mean;
        sum_sq += diff * diff;
    }
    (sum_sq / (n as f64 - 1.0)).sqrt()
}

fn date_to_ts(date: NaiveDate) -> Result<i64> {
    date.and_hms_opt(0, 0, 0)
        .map(|dt| dt.and_utc().timestamp())
        .ok_or_else(|| anyhow!("invalid date for timestamp"))
}

fn classify(actual: f64, median: f64) -> &'static str {
    if actual < (1.0 - THRESH) * median {
        return "BUY";
    }
    if actual > (1.0 + THRESH) * median {
        return "SELL";
    }
    "HOLD"
}

/// 1234567.891 -> "1,234,567.89" with `decimals` digits after the point.
fn fmt_num(v: f64, decimals: usize) -> String {
    let s = format!("{:.*}", decimals, v.abs());
    let (int, frac) = s.split_once('.').map_or((s.as_str(), ""), |(i, f)| (i, f));
    let mut out = String::new();
    for (i, c) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    if !frac.is_empty() {
        out.push('.');
        out.push_str(frac);
    }
    if v < 0.0 && out.chars().any(|c| c != '0' && c != '.' && c != ',') {
        out.insert(0, '-');
    }
    out
}

fn pct_class(pct: f64) -> &'static str {
    if pct > 0.0 {
        "up"
    } else if pct < 0.0 {
        "down"
    } else {
        "flat"
    }
}

fn weather(pct: f64) -> &'static str {
    match pct_class(pct) {
        "up" => "☀️",
        "down" => "☔",
        _ => "☁️",
    }
}

fn fund_card(f: &FundResult) -> String {
    let drawdown = (f.nav / f.high - 1.0) * 100.0;
    let Some(estimate) = f.estimate else {
        let diff = f.nav - f.prev.1;
        let change = (f.nav / f.prev.1 - 1.0) * 100.0;
        let badge = if f.nav >= f.high {
            r#"<span class="badge high">🎉 High today</span>"#
        } else {
            ""
        };
        return format!(
            r#"<article class="card" data-fund="{id}">
  <header><h3>{name}</h3>{badge}</header>
  <div class="row">
    <div><div class="label">NAV {date}</div><div class="num big {cls}">{nav}<small>円</small></div></div>
  </div>
  <div class="change {cls}">{weather} {diff}円 ({change:+.2}%) vs {prev_date}</div>
  <div class="label">Published. The next estimate comes with the 7:17 update.</div>
  {spark}
  <footer>High since inception {high}円 on {high_date} · {drawdown:+.1}%<br>{accuracy}</footer>
</article>"#,
            name = f.name,
            id = f.id,
            spark = sparkline(&f.spark),
            date = f.date.format("%m/%d"),
            nav = fmt_num(f.nav, 0),
            cls = pct_class(change),
            diff = format!("{}{}", if diff > 0.0 { "+" } else { "" }, fmt_num(diff, 0)),
            change = change,
            prev_date = f.prev.0.format("%m/%d"),
            weather = weather(change),
            high = fmt_num(f.high, 0),
            high_date = f.high_date,
            accuracy = accuracy_line(&f.accuracy),
        );
    };
    let target = format!("{} NAV estimate", f.target.format("%m/%d"));
    let change = (estimate / f.nav - 1.0) * 100.0;
    let badge = if estimate.round() > f.high {
        r#"<span class="badge high">🎉 New high (estimate)</span>"#.to_string()
    } else if f.nav >= f.high {
        format!(
            r#"<span class="badge high">🎉 High on {}</span>"#,
            f.date.format("%m/%d")
        )
    } else {
        String::new()
    };
    format!(
        r#"<article class="card" data-fund="{id}">
  <header><h3>{name}</h3>{badge}</header>
  <div class="row">
    <div><div class="label">NAV {date}</div><div class="num">{nav}<small>円</small></div></div>
    <div class="arrow">→</div>
    <div><div class="label">{target}</div><div class="num big {cls}">{est}<small>円</small></div></div>
  </div>
  <div class="change {cls}">{weather} {change:+.2}%</div>
  <div class="label">{attribution}</div>
  {spark}
  <footer>High since inception {high}円 on {high_date} · {drawdown:+.1}%<br>{accuracy}</footer>
</article>"#,
        name = f.name,
        id = f.id,
        spark = sparkline(&f.spark),
        attribution = f
            .attribution
            .as_ref()
            .map_or(String::new(), Attribution::line),
        target = target,
        badge = badge,
        date = f.date.format("%m/%d"),
        nav = fmt_num(f.nav, 0),
        est = fmt_num(estimate, 0),
        cls = pct_class(change),
        change = change,
        weather = weather(change),
        high = fmt_num(f.high, 0),
        high_date = f.high_date,
        drawdown = drawdown,
        accuracy = accuracy_line(&f.accuracy),
    )
}

/// A line through `navs` filling a 100×30 box, coloured by the move from first to last.
fn sparkline(navs: &[f64]) -> String {
    if navs.len() < 2 {
        return String::new();
    }
    let lo = navs.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = navs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (hi - lo).max(f64::EPSILON);
    let step = 100.0 / (navs.len() - 1) as f64;
    let points: Vec<String> = navs
        .iter()
        .enumerate()
        .map(|(i, v)| {
            format!(
                "{:.1},{:.1}",
                i as f64 * step,
                29.0 - (v - lo) / span * 28.0
            )
        })
        .collect();
    format!(
        r#"<svg class="spark {}" viewBox="0 0 100 30" preserveAspectRatio="none" aria-hidden="true"><polyline points="{}"/></svg>"#,
        pct_class(navs[navs.len() - 1] - navs[0]),
        points.join(" ")
    )
}

fn accuracy_line(a: &Accuracy) -> String {
    match a.last {
        None => "Estimate error: collecting".to_string(),
        Some((date, est, actual)) => format!(
            "Estimate error: avg {}円 ({:.2}%) over {} NAVs · after {}: {} vs actual {}",
            fmt_num(a.mae, 0),
            a.mape,
            a.count,
            date.format("%m/%d"),
            fmt_num(est, 0),
            fmt_num(actual, 0),
        ),
    }
}

fn asset_card(r: &AssetResult) -> String {
    let change = (r.median / r.actual - 1.0) * 100.0;
    format!(
        r#"<article class="card">
  <header><h3>{name}</h3><span class="badge {action_cls}">{action}</span></header>
  <div class="row">
    <div><div class="label">Now</div><div class="num">{actual}</div></div>
    <div class="arrow">→</div>
    <div><div class="label">{days}-day forecast</div><div class="num big {cls}">{median}</div></div>
  </div>
  <div class="change {cls}">{change:+.2}%</div>
  <footer>Range {lower} – {upper}</footer>
  <img src="{img}" alt="{name} forecast chart" loading="lazy">
</article>"#,
        name = r.name,
        action_cls = r.action.to_lowercase(),
        action = r.action,
        actual = fmt_num(r.actual, 2),
        days = FORECAST_DAYS,
        median = fmt_num(r.median, 2),
        cls = pct_class(change),
        change = change,
        lower = fmt_num(r.lower, 2),
        upper = fmt_num(r.upper, 2),
        img = urlencoding::encode(&chart_file(&r.name)),
    )
}

fn build_html(results: &[AssetResult], funds: &[FundResult]) -> String {
    let updated = (Utc::now() + Duration::hours(9)).format("%Y-%m-%d %H:%M JST");
    let mut body = String::new();
    let mut groups: Vec<&str> = funds.iter().map(|f| f.group).collect();
    groups.dedup();
    for group in groups {
        body.push_str(&format!(
            r#"<section><h2>{} <span>today's NAV estimates</span></h2><div class="grid">"#,
            group
        ));
        for f in funds.iter().filter(|f| f.group == group) {
            body.push_str(&fund_card(f));
        }
        body.push_str("</div></section>");
    }
    if !results.is_empty() {
        body.push_str(&format!(
            r#"<section><h2>Markets <span>{}-day forecast</span></h2><div class="grid">"#,
            FORECAST_DAYS
        ));
        for r in results {
            body.push_str(&asset_card(r));
        }
        body.push_str("</div></section>");
    }
    TEMPLATE
        .replace("{{updated}}", &updated.to_string())
        .replace("{{body}}", &body)
}

const TEMPLATE: &str = include_str!("template.html");

fn render_chart(name: &str, series: &[SeriesPoint], forecast: &[ForecastPoint]) -> Result<()> {
    if series.is_empty() {
        return Ok(());
    }

    let start_date = series.first().unwrap().date;
    let end_date = match forecast.last() {
        Some(p) => p.date,
        None => series.last().unwrap().date,
    };

    let mut y_min = series.iter().map(|p| p.close).fold(f64::INFINITY, f64::min);
    let mut y_max = series
        .iter()
        .map(|p| p.close)
        .fold(f64::NEG_INFINITY, f64::max);

    for p in forecast {
        y_min = y_min.min(p.lower);
        y_max = y_max.max(p.upper);
    }

    let padding = (y_max - y_min).abs() * 0.05;
    y_min -= padding;
    y_max += padding;

    let filename = format!("{}/{}", SITE_DIR, chart_file(name));
    let root = BitMapBackend::new(&filename, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(name, ("sans-serif", 36))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(start_date..end_date, y_min..y_max)?;

    chart.configure_mesh().x_labels(8).y_labels(10).draw()?;

    chart
        .draw_series(LineSeries::new(
            series.iter().map(|p| (p.date, p.close)),
            &RGBColor(30, 144, 255),
        ))?
        .label("History")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(30, 144, 255)));

    chart
        .draw_series(LineSeries::new(
            forecast.iter().map(|p| (p.date, p.median)),
            &RGBColor(220, 50, 32),
        ))?
        .label("Forecast")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(220, 50, 32)));

    chart.draw_series(LineSeries::new(
        forecast.iter().map(|p| (p.date, p.upper)),
        &RGBColor(200, 0, 0),
    ))?;

    chart.draw_series(LineSeries::new(
        forecast.iter().map(|p| (p.date, p.lower)),
        &RGBColor(200, 0, 0),
    ))?;

    if let Some(last) = series.last() {
        chart
            .draw_series(std::iter::once(Circle::new(
                (last.date, last.close),
                6,
                RGBColor(0, 153, 76).filled(),
            )))?
            .label("Actual")
            .legend(|(x, y)| Circle::new((x + 10, y), 5, RGBColor(0, 153, 76).filled()));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(&BLACK)
        .draw()?;

    root.present().context("write chart")?;
    Ok(())
}

fn chart_file(name: &str) -> String {
    format!("{}_forecast.png", name)
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::{
        Attribution, DateTime, NaiveDate, SeriesPoint, Utc, change_between, fmt_num, fund_id,
        history_file, json, nav_target, parse_ttm, record, sparkline,
    };

    use super::{
        Accuracy, FundResult, HISTORY_KEEP, accuracy_line, build_html, classify, etf_change,
        fund_card, fx_part, msci_change, now_rate, parse_msci, parse_nav, read_history,
    };

    fn day(s: &str) -> NaiveDate {
        s.parse().unwrap()
    }

    fn point(date: &str, close: f64) -> SeriesPoint {
        SeriesPoint {
            date: day(date),
            at: Utc::now(),
            close,
        }
    }

    /// A fund on NAV 100 (11/20, previous 98), high 110 on 11/02, nothing scored yet.
    fn fund(group: &'static str, name: &str, estimate: Option<f64>) -> FundResult {
        FundResult {
            group,
            name: name.to_string(),
            id: "F".to_string(),
            date: day("2026-11-20"),
            nav: 100.0,
            prev: (day("2026-11-19"), 98.0),
            target: day("2026-11-23"),
            estimate,
            attribution: estimate.map(|e| Attribution::Parts {
                legs: vec![("^GSPC", 1.0, e / 100.0)],
                fx: None,
            }),
            high_date: day("2026-11-02"),
            high: 110.0,
            spark: vec![98.0, 100.0],
            accuracy: Accuracy {
                count: 0,
                mae: 0.0,
                mape: 0.0,
                last: None,
            },
        }
    }

    // FUND-2
    #[test]
    fn parses_nav_csvs() {
        let csv = "\u{fffd}\u{fffd},\u{fffd}\u{fffd}\n2026/11/02,37188,100\n\
                   2026\u{fffd}11\u{fffd}04\u{fffd},37306.5,101\nnot,a row\n2026/13/01,1\n";
        assert_eq!(
            parse_nav(csv),
            vec![(day("2026-11-02"), 37188.0), (day("2026-11-04"), 37306.5)]
        );
    }

    // EST-4
    #[test]
    fn converts_at_the_ttm_once_it_is_out() {
        assert_eq!(now_rate(Ok(Some(158.0)), || 157.5), 158.0);
        assert_eq!(now_rate(Ok(None), || 157.5), 157.5);
        assert_eq!(now_rate(Err(anyhow::anyhow!("down")), || 157.5), 157.5);
        let daily = [point("2026-11-19", 150.0), point("2026-11-20", 153.0)];
        assert!(
            (fx_part(Some(150.0), 151.5, &daily, day("2026-11-20")).unwrap() - 1.01).abs() < 1e-12
        );
        // Without the NAV date's TTM: latest daily close / last close before the NAV date.
        assert_eq!(
            fx_part(None, 151.5, &daily, day("2026-11-20")).unwrap(),
            1.02
        );
    }

    // EST-6
    #[test]
    fn follows_the_tokyo_etf_once_it_has_traded() {
        let before = [point("2026-11-19", 100.0), point("2026-11-20", 102.0)];
        assert_eq!(etf_change(&before, day("2026-11-20")).unwrap(), None);
        let after = [point("2026-11-20", 102.0), point("2026-11-21", 104.04)];
        assert!((etf_change(&after, day("2026-11-20")).unwrap().unwrap() - 1.02).abs() < 1e-12);
    }

    // NAV-1, FUND-1, FRESH-1
    #[test]
    fn shows_funds_in_their_groups() {
        let funds = [
            fund("MSCI", "A", Some(101.0)),
            fund("S&P500", "B", Some(101.0)),
            fund("MSCI", "C", Some(101.0)),
        ];
        let html = build_html(&[], &funds);
        let at = |t: &str| html.find(t).unwrap();
        assert!(
            at("<h2>MSCI ") < at(">A<") && at(">A<") < at(">C<") && at(">C<") < at("<h2>S&P500 ")
        );
        let updated = &html[at("Updated ")..];
        assert!(updated[8..].starts_with("20") && updated[24..].starts_with(" JST"));
    }

    // NAV-1, NAV-2, NAV-3, ATTR-1
    #[test]
    fn shows_the_estimate_and_its_change() {
        let up = fund_card(&fund("G", "F", Some(102.0)));
        assert!(up.contains("NAV 11/20") && up.contains("11/23 NAV estimate"));
        assert!(up.contains(r#"class="change up">☀️ +2.00%"#));
        assert!(up.contains("S&P500 +2.00% · USD/JPY hedged"));
        assert!(
            fund_card(&fund("G", "F", Some(97.0))).contains(r#"class="change down">☔ -3.00%"#)
        );
        assert!(
            fund_card(&fund("G", "F", Some(100.0))).contains(r#"class="change flat">☁️ +0.00%"#)
        );
    }

    // NAV-4, ATTR-1
    #[test]
    fn shows_a_published_nav() {
        let card = fund_card(&fund("G", "F", None));
        assert!(card.contains(r#"class="change up">☀️ +2円 (+2.04%) vs 11/19"#));
        assert!(card.contains("next estimate comes with the 7:17 update"));
        assert!(!card.contains("USD/JPY"));
    }

    // ACC-1, ACC-2
    #[test]
    fn shows_the_high_and_drawdown() {
        let card = fund_card(&fund("G", "F", Some(102.0)));
        assert!(card.contains("High since inception 110円 on 2026-11-02 · -9.1%"));
        assert!(!card.contains("🎉"));
        assert!(fund_card(&fund("G", "F", Some(111.0))).contains("🎉 New high (estimate)"));
        let mut at_high = fund("G", "F", Some(99.0));
        at_high.high = 100.0;
        assert!(fund_card(&at_high).contains("🎉 High on 11/20"));
        at_high.estimate = None;
        assert!(fund_card(&at_high).contains("🎉 High today"));
    }

    // ACC-3
    #[test]
    fn shows_the_estimate_error() {
        let mut a = fund("G", "F", None).accuracy;
        assert_eq!(accuracy_line(&a), "Estimate error: collecting");
        a.count = 12;
        a.mae = 70.4;
        a.mape = 0.19;
        a.last = Some((day("2026-11-19"), 35558.0, 35574.0));
        let line = accuracy_line(&a);
        assert!(
            line.starts_with("Estimate error: avg 70円 (0.19%) over 12 NAVs"),
            "{line}"
        );
        assert!(line.contains("35,558") && line.contains("35,574"), "{line}");
    }

    // MKT-2
    #[test]
    fn classifies_markets() {
        assert_eq!(classify(97.9, 100.0), "BUY");
        assert_eq!(classify(98.0, 100.0), "HOLD");
        assert_eq!(classify(102.0, 100.0), "HOLD");
        assert_eq!(classify(102.1, 100.0), "SELL");
    }

    // OPS-2
    #[test]
    fn reads_the_published_history() {
        assert_eq!(read_history(404, "Not Found").unwrap(), json!({}));
        assert_eq!(read_history(200, r#"{"F": []}"#).unwrap(), json!({"F": []}));
        assert!(read_history(503, "").is_err());
        assert!(read_history(200, "<html>").is_err());
    }

    // OPS-3
    #[test]
    fn keeps_at_most_2500_entries() {
        let start = day("2010-01-01");
        let list: Vec<_> = (0..HISTORY_KEEP as i64 + 5)
            .map(|i| json!({"base": (start + chrono::Duration::days(i)).to_string(), "estimate": 1.0, "actual": 1.0}))
            .collect();
        let mut h = json!({ "F": list });
        let rows = vec![(day("2026-11-20"), 1.0)];
        record(
            &mut h,
            "F",
            &rows,
            None,
            "2026-11-20T00:00:00Z".parse().unwrap(),
        );
        assert_eq!(h["F"].as_array().unwrap().len(), HISTORY_KEEP);
        assert_eq!(
            h["F"][0]["base"],
            (start + chrono::Duration::days(5)).to_string()
        );
    }

    // ATTR-2 to ATTR-6
    #[test]
    fn attributes_estimates() {
        let one = Attribution::Parts {
            legs: vec![("^GSPC", 1.0, 0.9679)],
            fx: Some(0.9908),
        };
        assert_eq!(one.line(), "S&P500 -3.21% · USD/JPY -0.92%");
        assert!((one.ratio() - 0.9679 * 0.9908).abs() < 1e-12);

        let lev = Attribution::Parts {
            legs: vec![("^NDX", 3.0, 0.985)],
            fx: None,
        };
        assert_eq!(lev.line(), "NASDAQ100 -1.50% ×3 · USD/JPY hedged");
        assert!((lev.ratio() - 0.955).abs() < 1e-12);

        let two = Attribution::Parts {
            legs: vec![("ACWI", 1.0, 1.004), ("GLDM", 1.0, 1.011)],
            fx: Some(1.0),
        };
        assert_eq!(
            two.line(),
            "MSCI ACWI +0.40% · Gold +1.10% · USD/JPY +0.00%"
        );
        assert!((two.ratio() - 1.015).abs() < 1e-12);

        let etf = Attribution::Etf("1540.T", 1.0085);
        assert_eq!(etf.line(), "1540 (yen) +0.85%");
        assert_eq!(etf.ratio(), 1.0085);
    }

    // HIST-7
    #[test]
    fn writes_fund_history() {
        let d = |s: &str| s.parse::<NaiveDate>().unwrap();
        let rows = [
            (d("2026-11-02"), 100.0),
            (d("2026-11-04"), 110.0),
            (d("2026-11-05"), 99.0),
            (d("2026-11-06"), 104.5),
        ];
        let scored = [
            json!({"base": "2026-10-30", "estimate": 101.0, "actual": 100.0}),
            json!({"base": "2026-11-02", "estimate": 111.0, "actual": 110.0}),
            json!({"base": "2026-11-04", "estimate": null, "actual": 99.0}),
            json!({"base": "2026-11-05", "runs": [["2026-11-05T22:17:00Z", 105.0]], "estimate": null, "actual": null}),
        ];
        let f = history_file(&rows, &scored);
        assert_eq!(f["fields"][2], "estimated_nav");
        assert_eq!(
            f["rows"][0],
            json!(["2026-11-02", 100.0, 101.0, 1.0, 1.0, 0.0])
        );
        assert_eq!(
            f["rows"][1],
            json!(["2026-11-04", 110.0, 111.0, 1.0, 0.909, 0.0])
        );
        assert_eq!(
            f["rows"][2],
            json!(["2026-11-05", 99.0, null, null, null, -10.0])
        );
        assert_eq!(
            f["rows"][3],
            json!(["2026-11-06", 104.5, null, null, null, -5.0])
        );
    }

    // HIST-8: estimates outlive the scoring window.
    #[test]
    fn keeps_estimates_past_the_score_window() {
        let d = |s: &str| s.parse::<NaiveDate>().unwrap();
        let start = d("2020-01-01");
        // 130 scored NAVs: the 10 oldest were 10 off, the rest exact.
        let list: Vec<_> = (0..130)
            .map(|i| {
                let base = (start + chrono::Duration::days(i)).to_string();
                let est = if i < 10 { 110.0 } else { 100.0 };
                json!({"base": base, "estimate": est, "actual": 100.0})
            })
            .collect();
        let mut h = json!({ "F": list });
        let rows = vec![(start + chrono::Duration::days(130), 100.0)];
        let a = record(
            &mut h,
            "F",
            &rows,
            None,
            "2026-11-04T10:53:00Z".parse().unwrap(),
        );
        assert_eq!(h["F"].as_array().unwrap().len(), 130);
        assert_eq!(a.count, 120);
        assert_eq!(a.mae, 0.0);
    }

    #[test]
    fn names_funds_by_code() {
        assert_eq!(
            fund_id("https://www.am.mufg.jp/fund_file/setteirai/253425.csv"),
            "253425"
        );
        assert_eq!(
            fund_id("https://www.rakuten-toushin.co.jp/assets/csv/chart_100086.csv"),
            "chart_100086"
        );
        assert_eq!(
            fund_id("https://toushin-lib.fwg.ne.jp/x?isinCd=JP90C000J569&associFundCd=89311199"),
            "JP90C000J569"
        );
    }

    // HIST-1
    #[test]
    fn draws_sparklines() {
        let s = sparkline(&[100.0, 90.0, 95.0]);
        assert!(s.contains(r#"class="spark down""#));
        assert!(s.contains(r#"points="0.0,1.0 50.0,29.0 100.0,15.0""#));
        assert!(sparkline(&[5.0, 5.0]).contains("spark flat"));
        assert_eq!(sparkline(&[1.0]), "");
    }

    #[test]
    fn records_and_scores_estimates() {
        let d = |s: &str| s.parse::<NaiveDate>().unwrap();
        let t = |s: &str| s.parse::<DateTime<Utc>>().unwrap();
        let mut h = json!({});
        // 11/03 is a holiday. Runs on 11/03 evening and 11/04 morning, noon and evening.
        let rows = vec![(d("2026-10-30"), 100.0), (d("2026-11-02"), 101.0)];
        assert_eq!(
            record(&mut h, "F", &rows, Some(102.0), t("2026-11-03T10:17:00Z")).count,
            0
        );
        record(&mut h, "F", &rows, Some(103.0), t("2026-11-03T22:17:00Z"));
        record(&mut h, "F", &rows, Some(104.0), t("2026-11-04T04:17:00Z"));
        // After 15:30 JST: the 11/04 NAV is already priced, so this run must not be scored.
        record(&mut h, "F", &rows, Some(109.0), t("2026-11-04T10:17:00Z"));
        // The 11/04 NAV (103.5) is out; the noon run's 104.0 is scored.
        let rows = [rows, vec![(d("2026-11-04"), 103.5)]].concat();
        let a = record(&mut h, "F", &rows, Some(110.0), t("2026-11-04T10:53:00Z"));
        assert_eq!(a.count, 1);
        assert_eq!(a.mae, 0.5);
        assert_eq!(a.last, Some((d("2026-11-02"), 104.0, 103.5)));
        assert_eq!(h["F"].as_array().unwrap().len(), 2);
        assert!(h["F"][0].get("runs").is_none());
    }

    #[test]
    fn drops_old_pending_entries() {
        let d = |s: &str| s.parse::<NaiveDate>().unwrap();
        let mut h = json!({"F": [
            {"base": "2026-10-30", "estimate": 100.0, "actual": 100.5},
            {"base": "2026-11-02", "estimate": 109.0, "actual": null},
        ]});
        let rows = vec![(d("2026-11-02"), 101.0), (d("2026-11-04"), 103.5)];
        let a = record(
            &mut h,
            "F",
            &rows,
            Some(104.0),
            "2026-11-04T10:53:00Z".parse().unwrap(),
        );
        assert_eq!(a.count, 1);
        assert_eq!(h["F"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn targets_todays_nav() {
        let d = |s: &str| s.parse::<NaiveDate>().unwrap();
        let t = |s: &str| s.parse::<DateTime<Utc>>().unwrap();
        // Wed 07:17 JST is still Tue in UTC; Sat and Sun target Monday.
        assert_eq!(nav_target(t("2026-11-03T22:17:00Z")), d("2026-11-04"));
        assert_eq!(nav_target(t("2026-11-07T01:00:00Z")), d("2026-11-09"));
        assert_eq!(nav_target(t("2026-11-08T22:17:00Z")), d("2026-11-09"));
    }

    #[test]
    fn ignores_closes_after_the_target() {
        let d = |s: &str| s.parse::<NaiveDate>().unwrap();
        let p = |s: &str, close| SeriesPoint {
            date: d(s),
            at: Utc::now(),
            close,
        };
        // NAV 11/02 is priced at the 10/30 close, the 11/04 NAV at the 11/03 close; 11/04 is later.
        let series = [
            p("2026-10-30", 100.0),
            p("2026-11-02", 101.0),
            p("2026-11-03", 102.0),
            p("2026-11-04", 110.0),
        ];
        let r = change_between(&series, d("2026-11-02"), d("2026-11-04")).unwrap();
        assert_eq!(r, 1.02);
    }

    #[test]
    fn follows_msci_then_the_etf() {
        // EST-9
        let d = |s: &str| s.parse::<NaiveDate>().unwrap();
        let p = |s: &str, close| SeriesPoint {
            date: d(s),
            at: Utc::now(),
            close,
        };
        let msci = parse_msci(
            r#"{"indexes":{"INDEX_LEVELS":[{"level_eod":100.0,"calc_date":20261123},{"level_eod":101.0,"calc_date":20261124}]}}"#,
        )
        .unwrap();
        let etf = [
            p("2026-11-23", 50.0),
            p("2026-11-24", 49.0),
            p("2026-11-25", 49.49),
        ];
        // NAV 11/24 to 11/25: MSCI's 11/24 level is out, so the ETF's -2% doesn't count.
        let r = msci_change(&msci, &etf, d("2026-11-24"), d("2026-11-25")).unwrap();
        assert!((r - 1.01).abs() < 1e-9);
        // To 11/26 before MSCI has 11/25: MSCI to 11/24, then the ETF's +1% on 11/25.
        let r = msci_change(&msci, &etf, d("2026-11-24"), d("2026-11-26")).unwrap();
        assert!((r - 1.01 * 1.01).abs() < 1e-9);
    }

    #[test]
    fn parses_ttm() {
        let d = |s: &str| s.parse::<NaiveDate>().unwrap();
        let html = r#"<h2>2026年11月4日現在　As of November 4, 2026</h2>
            <tr><td>Euro</td></tr>
            <tr><td>US Dollar</td><td class="t_center">USD</td>
            <td class="t_right">159.59 </td><td class="t_right">157.59 </td></tr>"#;
        assert!((parse_ttm(html, d("2026-11-04")).unwrap() - 158.59).abs() < 1e-9);
        assert_eq!(parse_ttm(html, d("2026-11-05")), None);
    }

    #[test]
    fn formats_numbers() {
        assert_eq!(fmt_num(37188.0, 0), "37,188");
        assert_eq!(fmt_num(1234567.891, 2), "1,234,567.89");
        assert_eq!(fmt_num(999.0, 0), "999");
        assert_eq!(fmt_num(-1234.5, 1), "-1,234.5");
        assert_eq!(fmt_num(-0.001, 2), "0.00");
    }
}
