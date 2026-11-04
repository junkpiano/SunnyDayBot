use anyhow::{Context, Result, anyhow};
use augurs::prophet::wasmstan::WasmstanOptimizer;
use augurs::prophet::{PredictionData, Prophet, TrainingData};
use chrono::{DateTime, Duration, NaiveDate, Utc};
use plotters::prelude::*;
use serde_json::{Map, Value, json};
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
    /// Yahoo ticker of what the fund holds.
    ticker: &'static str,
    /// Priced at the NAV date's own Tokyo close (a fund holding a Tokyo-listed ETF), so the
    /// estimate measures `ticker` and USD/JPY from 15:30 JST that day instead of from the close
    /// before it.
    tokyo_close: bool,
}

// am.mufg.jp's own CSV answers 403 to GitHub's runners, so MUFG funds come from the
// Investment Trusts Association's library (ISIN and association fund code).
// ponytail: distribution funds drop by the payout on their record date; the estimate misses that day.
const FUNDS: &[Fund] = &[
    Fund {
        group: "All-Country",
        name: "eMAXIS Slim 全世界株式（オール・カントリー）",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000H1T1&associFundCd=0331418A",
        ticker: "ACWI",
        tokyo_close: false,
    },
    Fund {
        group: "All-Country",
        name: "楽天・プラス・オールカントリー株式インデックス・ファンド",
        url: "https://www.rakuten-toushin.co.jp/assets/csv/chart_100086.csv",
        ticker: "ACWI",
        tokyo_close: false,
    },
    Fund {
        group: "S&P500",
        name: "eMAXIS Slim 米国株式（S&P500）",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000GKC6&associFundCd=03311187",
        ticker: "^GSPC",
        tokyo_close: false,
    },
    Fund {
        group: "S&P500",
        name: "SBI・V・S&P500インデックス・ファンド",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000J569&associFundCd=89311199",
        ticker: "^GSPC",
        tokyo_close: false,
    },
    Fund {
        group: "S&P500",
        name: "楽天・プラス・S&P500インデックス・ファンド",
        url: "https://www.rakuten-toushin.co.jp/assets/csv/chart_100087.csv",
        ticker: "^GSPC",
        tokyo_close: false,
    },
    Fund {
        group: "Gold",
        name: "SBI・iシェアーズ・ゴールドファンド（為替ヘッジなし）",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000PLM4&associFundCd=8931A236",
        ticker: "IGLN.L",
        tokyo_close: false,
    },
    Fund {
        group: "Gold",
        name: "三菱UFJ 純金ファンド",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C0007G10&associFundCd=03311112",
        ticker: "GC=F",
        tokyo_close: true,
    },
    Fund {
        group: "Gold",
        name: "楽天・ゴールド・ファンド（為替ヘッジなし）",
        url: "https://www.rakuten-toushin.co.jp/assets/csv/chart_100123.csv",
        ticker: "GLDM",
        tokyo_close: false,
    },
    Fund {
        group: "Gold",
        name: "ステート・ストリート・ゴールド・オープン（為替ヘッジなし）",
        url: "https://toushin-lib.fwg.ne.jp/FdsWeb/FDST030000/csv-file-download?isinCd=JP90C000RXU8&associFundCd=55311258",
        ticker: "GLDM",
        tokyo_close: false,
    },
    Fund {
        group: "SCHD",
        name: "楽天・シュワブ・高配当株式・米国ファンド（資産成長型）",
        url: "https://www.rakuten-toushin.co.jp/assets/csv/chart_100119.csv",
        ticker: "SCHD",
        tokyo_close: false,
    },
    Fund {
        group: "SCHD",
        name: "楽天・シュワブ・高配当株式・米国ファンド（四半期決算型）",
        url: "https://www.rakuten-toushin.co.jp/assets/csv/chart_100105.csv",
        ticker: "SCHD",
        tokyo_close: false,
    },
];
const USDJPY: &str = "JPY=X";

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
    estimate: f64,
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

        let (forecast, actual_price) = forecast_series(&series)?;
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

/// Each fund's next NAV estimate; funds whose data is unavailable are left out.
fn estimate_funds(history: &mut Value) -> Vec<FundResult> {
    let fx = match fetch_series(USDJPY) {
        Ok(fx) => fx,
        Err(err) => {
            eprintln!("Failed to fetch USD/JPY: {:#}", err);
            return Vec::new();
        }
    };

    let mut funds = Vec::new();
    for f in FUNDS {
        let name = f.name;
        let fund = (|| -> Result<FundResult> {
            let rows = fetch_nav(f.url)?;
            let (date, nav) = *rows.last().unwrap();
            // The latest of equal highs, so a fund back at its high shows today's date.
            let (high_date, high) = rows
                .iter()
                .copied()
                .fold(rows[0], |h, r| if r.1 >= h.1 { r } else { h });
            let ratio = if f.tokyo_close {
                // 15:30 JST. ponytail: hourly bars, so the base is up to an hour after the close.
                let close = date.and_hms_opt(6, 30, 0).unwrap().and_utc();
                change_since_time(&fetch_chart(f.ticker, "60m", "1mo")?, close)?
                    * change_since_time(&fetch_chart(USDJPY, "60m", "1mo")?, close)?
            } else {
                change_since(&fetch_series(f.ticker)?, date)? * change_since(&fx, date)?
            };
            let estimate = nav * ratio;
            Ok(FundResult {
                group: f.group,
                name: name.to_string(),
                date,
                nav,
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
    estimate: f64,
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

    let base_str = base.to_string();
    let run = json!([now.to_rfc3339(), estimate]);
    match list.iter_mut().find(|e| e["base"] == base_str.as_str()) {
        Some(e) => e["runs"].as_array_mut().unwrap().push(run),
        None => {
            list.push(json!({"base": base_str, "runs": [run], "estimate": null, "actual": null}))
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

fn fund_card(f: &FundResult) -> String {
    let change = (f.estimate / f.nav - 1.0) * 100.0;
    let drawdown = (f.nav / f.high - 1.0) * 100.0;
    let badge = if f.estimate.round() > f.high {
        r#"<span class="badge high">🎉 New high (estimate)</span>"#
    } else if f.nav >= f.high {
        r#"<span class="badge high">🎉 High today</span>"#
    } else {
        ""
    };
    format!(
        r#"<article class="card">
  <header><h3>{name}</h3>{badge}</header>
  <div class="row">
    <div><div class="label">NAV {date}</div><div class="num">{nav}<small>円</small></div></div>
    <div class="arrow">→</div>
    <div><div class="label">Next estimate</div><div class="num big {cls}">{est}<small>円</small></div></div>
  </div>
  <div class="change {cls}">{change:+.2}%</div>
  <footer>High since inception {high}円 on {high_date} · {drawdown:+.1}%<br>{accuracy}</footer>
</article>"#,
        name = f.name,
        badge = badge,
        date = f.date.format("%m/%d"),
        nav = fmt_num(f.nav, 0),
        est = fmt_num(f.estimate, 0),
        cls = pct_class(change),
        change = change,
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
            r#"<section><h2>{} <span>next NAV estimate</span></h2><div class="grid">"#,
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
    use super::{DateTime, NaiveDate, Utc, fmt_num, json, record};

    #[test]
    fn records_and_scores_estimates() {
        let d = |s: &str| s.parse::<NaiveDate>().unwrap();
        let t = |s: &str| s.parse::<DateTime<Utc>>().unwrap();
        let mut h = json!({});
        // 11/03 is a holiday. Runs on 11/03 evening and 11/04 morning, noon and evening.
        let rows = vec![(d("2026-10-30"), 100.0), (d("2026-11-02"), 101.0)];
        assert_eq!(
            record(&mut h, "F", &rows, 102.0, t("2026-11-03T10:17:00Z")).count,
            0
        );
        record(&mut h, "F", &rows, 103.0, t("2026-11-03T22:17:00Z"));
        record(&mut h, "F", &rows, 104.0, t("2026-11-04T04:17:00Z"));
        // After 15:30 JST: the 11/04 NAV is already priced, so this run must not be scored.
        record(&mut h, "F", &rows, 109.0, t("2026-11-04T10:17:00Z"));
        // The 11/04 NAV (103.5) is out; the noon run's 104.0 is scored.
        let rows = [rows, vec![(d("2026-11-04"), 103.5)]].concat();
        let a = record(&mut h, "F", &rows, 110.0, t("2026-11-04T10:53:00Z"));
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
            104.0,
            "2026-11-04T10:53:00Z".parse().unwrap(),
        );
        assert_eq!(a.count, 1);
        assert_eq!(h["F"].as_array().unwrap().len(), 2);
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
