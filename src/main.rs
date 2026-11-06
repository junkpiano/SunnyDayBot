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
const HISTORY_KEEP: usize = 120;
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
    date: NaiveDate,
    nav: f64,
    /// The NAV before `nav`.
    prev: (NaiveDate, f64),
    /// The NAV date estimated: today in Japan, or the next weekday on a weekend.
    target: NaiveDate,
    /// None once the `target` NAV is published: the one after it needs closes that haven't happened.
    estimate: Option<f64>,
    high_date: NaiveDate,
    high: f64,
    accuracy: Accuracy,
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

    // Stop rather than publish an empty history over the real one.
    let mut history = load_history().context("load the published history")?;
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
    let fx_now = match usd_ttm(target) {
        Ok(Some(ttm)) => ttm,
        Ok(None) => fx.last().map_or(f64::NAN, |p| p.close),
        Err(err) => {
            eprintln!("Failed to fetch today's TTM: {:#}", err);
            fx.last().map_or(f64::NAN, |p| p.close)
        }
    };
    let mut ttms: HashMap<NaiveDate, Option<f64>> = HashMap::new();

    // Many funds share an index; fetch each once.
    let mut cache: HashMap<&str, Vec<SeriesPoint>> = HashMap::new();
    let mut funds = Vec::new();
    for f in FUNDS {
        let name = f.name;
        let fund = (|| -> Result<FundResult> {
            let rows = fetch_nav(f.url)?;
            let (date, nav) = *rows.last().unwrap();
            let prev = rows[rows.len().saturating_sub(2)];
            // The latest of equal highs, so a fund back at its high shows today's date.
            let (high_date, high) = rows
                .iter()
                .copied()
                .fold(rows[0], |h, r| if r.1 >= h.1 { r } else { h });
            if date >= target {
                return Ok(FundResult {
                    group: f.group,
                    name: name.to_string(),
                    date,
                    nav,
                    prev,
                    target,
                    estimate: None,
                    high_date,
                    high,
                    accuracy: record(history, name, &rows, None, Utc::now()),
                });
            }
            let ratio = if let Some(etf) = f.tokyo_etf {
                let etf = fetch_series(etf)?;
                if etf.last().is_some_and(|p| p.date > date) {
                    // Change since the ETF's close on the NAV date.
                    change_since(&etf, date.succ_opt().unwrap())?
                } else {
                    // Tokyo hasn't traded since. 15:30 JST. ponytail: hourly bars, so the base
                    // is up to an hour after the close.
                    let close = date.and_hms_opt(6, 30, 0).unwrap().and_utc();
                    change_since_time(&fetch_chart(f.legs[0].0, "60m", "1mo")?, close)?
                        * change_since_time(&fetch_chart(USDJPY, "60m", "1mo")?, close)?
                }
            } else {
                // Closes before the target date: later ones belong to the NAV after it.
                let mut ratio = 1.0;
                for &(ticker, weight) in f.legs {
                    let series = match cache.get(ticker) {
                        Some(series) => series,
                        None => cache.entry(ticker).or_insert(fetch_series(ticker)?),
                    };
                    ratio += weight * (change_between(series, date, target)? - 1.0);
                }
                if !f.hedged {
                    // The NAV was converted at that day's TTM.
                    let base = *ttms.entry(date).or_insert_with(|| {
                        usd_ttm(date)
                            .inspect_err(|e| eprintln!("Failed to fetch TTM for {}: {:#}", date, e))
                            .ok()
                            .flatten()
                    });
                    ratio *= match base {
                        Some(base) => fx_now / base,
                        None => change_since(&fx, date)?,
                    };
                }
                ratio
            };
            let estimate = Some(nav * ratio);
            Ok(FundResult {
                group: f.group,
                name: name.to_string(),
                date,
                nav,
                prev,
                target,
                estimate,
                high_date,
                high,
                accuracy: record(history, name, &rows, estimate, Utc::now()),
            })
        })();
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
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(json!({}));
    }
    let text = resp.error_for_status()?.text()?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", url))
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

    score(list)
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
    let rows: Vec<(NaiveDate, f64)> = String::from_utf8_lossy(&bytes)
        .lines()
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
        .collect();
    if rows.is_empty() {
        return Err(anyhow!("no NAV rows in csv {}", url));
    }
    Ok(rows)
}

/// Latest close / last close dated before `nav_date`. A NAV for Japan date D is priced at the
/// foreign closes of the day before D and at D's morning USD/JPY, which is the close before D.
// ponytail: ignores ETF ex-dividend drops (ACWI, twice a year) and Japan-stock timing; correct for them if the error matters.
/// Latest close / close of the last bar that started before `at`.
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
            r#"<article class="card">
  <header><h3>{name}</h3>{badge}</header>
  <div class="row">
    <div><div class="label">NAV {date}</div><div class="num big {cls}">{nav}<small>円</small></div></div>
  </div>
  <div class="change {cls}">{weather} {diff}円 ({change:+.2}%) vs {prev_date}</div>
  <div class="label">Published. The next estimate comes with the 7:17 update.</div>
  <footer>High since inception {high}円 on {high_date} · {drawdown:+.1}%<br>{accuracy}</footer>
</article>"#,
            name = f.name,
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
        r#"<article class="card">
  <header><h3>{name}</h3>{badge}</header>
  <div class="row">
    <div><div class="label">NAV {date}</div><div class="num">{nav}<small>円</small></div></div>
    <div class="arrow">→</div>
    <div><div class="label">{target}</div><div class="num big {cls}">{est}<small>円</small></div></div>
  </div>
  <div class="change {cls}">{weather} {change:+.2}%</div>
  <footer>High since inception {high}円 on {high_date} · {drawdown:+.1}%<br>{accuracy}</footer>
</article>"#,
        name = f.name,
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
        DateTime, NaiveDate, SeriesPoint, Utc, change_between, fmt_num, json, nav_target,
        parse_ttm, record,
    };

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
