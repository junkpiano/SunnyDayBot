use anyhow::{anyhow, Context, Result};
use augurs::prophet::wasmstan::WasmstanOptimizer;
use augurs::prophet::{PredictionData, Prophet, TrainingData};
use chrono::{DateTime, Duration, NaiveDate, Utc};
use plotters::prelude::*;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::time::Duration as StdDuration;

const HISTORY_YEARS: i64 = 5;
const FORECAST_DAYS: usize = 30;
const TRAIN_DAYS: usize = 180;
const THRESH: f64 = 0.02;
const HTTP_RETRIES: usize = 2;
const HTTP_BACKOFF_SECS: u64 = 2;

const TICKERS: &[(&str, &str)] = &[
    ("^IXIC", "NASDAQ"),
    ("^GSPC", "S&P500"),
    ("^N225", "Nikkei225"),
];

#[derive(Clone, Copy)]
struct SeriesPoint {
    date: NaiveDate,
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

fn main() -> Result<()> {
    let docs_dir = Path::new("docs");
    if !docs_dir.exists() {
        fs::create_dir_all(docs_dir).context("create docs directory")?;
    }

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

    if results.is_empty() {
        println!("No forecasts were produced (all data unavailable).");
        return Ok(());
    }

    let md = build_markdown(&results);
    println!("\n{}\n", md);

    let mut index = String::new();
    index.push_str("# Forecast Summary\n\n");
    index.push_str(&md);
    index.push_str("\n\n## Forecast Charts\n\n");
    for result in &results {
        let image_path = format!("{}_forecast.png", result.name);
        index.push_str(&format!("### {}\n\n", result.name));
        index.push_str(&format!("![{} Forecast](./{})\n\n", result.name, image_path));
    }

    fs::write(docs_dir.join("index.md"), index).context("write docs/index.md")?;

    Ok(())
}

fn fetch_series(ticker: &str) -> Result<Vec<SeriesPoint>> {
    let now = Utc::now().date_naive();
    let start = now - Duration::days(365 * HISTORY_YEARS);

    let encoded = urlencoding::encode(ticker);
    let url = format!(
        "https://query1.finance.yahoo.com/v8/finance/chart/{}?interval=1d&range=5y",
        encoded
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
        let text = response.text().unwrap_or_else(|_| "<failed to read body>".to_string());

        if !status.is_success() {
            let err = anyhow!(
                "yahoo finance returned {} for {}: {}",
                status,
                ticker,
                text
            );
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

        let resp: Value =
            serde_json::from_str(&text).context("parse yahoo finance json")?;

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

            series.push(SeriesPoint { date, close });
        }

        return Ok(series);
    }

    Err(last_error.unwrap_or_else(|| anyhow!("request failed for {}", ticker)))
}

fn forecast_series(series: &[SeriesPoint]) -> Result<(Vec<ForecastPoint>, f64)> {
    let len = series.len();
    if len < 2 {
        return Err(anyhow!("not enough data to forecast"));
    }

    let actual_price = series[len - 1].close;

    let start = if len > TRAIN_DAYS { len - TRAIN_DAYS } else { 0 };
    let train = &series[start..];

    let mut ds: Vec<i64> = Vec::with_capacity(train.len());
    let mut y: Vec<f64> = Vec::with_capacity(train.len());
    for p in train {
        ds.push(date_to_ts(p.date)?);
        y.push(p.close);
    }

    let data = TrainingData::new(ds, y.clone()).context("create prophet training data")?;
    let mut prophet = Prophet::new(Default::default(), WasmstanOptimizer::new());
    prophet.fit(data, Default::default()).context("fit prophet model")?;

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

fn build_markdown(results: &[AssetResult]) -> String {
    let mut md = String::new();
    md.push_str("| Asset | ActualPrice | ForecastMedian | ForecastUpper | ForecastLower | Action |\n");
    md.push_str("| --- | --- | --- | --- | --- | --- |\n");

    for r in results {
        md.push_str(&format!(
            "| {} | {:.2} | {:.2} | {:.2} | {:.2} | {} |\n",
            r.name, r.actual, r.median, r.upper, r.lower, r.action
        ));
    }

    md
}

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
    let mut y_max = series.iter().map(|p| p.close).fold(f64::NEG_INFINITY, f64::max);

    for p in forecast {
        y_min = y_min.min(p.lower);
        y_max = y_max.max(p.upper);
    }

    let padding = (y_max - y_min).abs() * 0.05;
    y_min -= padding;
    y_max += padding;

    let filename = format!("docs/{}_forecast.png", name);
    let root = BitMapBackend::new(&filename, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(name, ("sans-serif", 36))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(start_date..end_date, y_min..y_max)?;

    chart
        .configure_mesh()
        .x_labels(8)
        .y_labels(10)
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            series.iter().map(|p| (p.date, p.close)),
            &RGBColor(30, 144, 255),
        ))?
        .label("History")
        .legend(|(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(30, 144, 255))
        });

    chart
        .draw_series(LineSeries::new(
            forecast.iter().map(|p| (p.date, p.median)),
            &RGBColor(220, 50, 32),
        ))?
        .label("Forecast")
        .legend(|(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(220, 50, 32))
        });

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
            .legend(|(x, y)| {
                Circle::new((x + 10, y), 5, RGBColor(0, 153, 76).filled())
            });
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(&BLACK)
        .draw()?;

    root.present().context("write chart")?;
    Ok(())
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}
