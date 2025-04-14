# alert_if_out_of_bounds.py (includes BTC + summary table with names and currency symbols)

import yfinance as yf
from prophet import Prophet
import pandas as pd
from datetime import datetime
from pathlib import Path

# === Configuration ===
TICKERS = [
    "^IXIC", "^GSPC", "VXUS", "JEPI", "JEPQ", "HDV", "SCHD", "VYM",
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "NVDA", "TSLA",
    "4755.T", "9432.T", "9434.T", "7203.T",
    "BTC-USD", "GC=F"
]
DAYS_FORWARD = 30

results = []  # Store summary info here

def format_currency(ticker, value):
    if ticker.endswith(".T") or ticker.endswith(".N"):
        return f"¥{value:,.2f}"
    else:
        return f"${value:,.2f}"

def analyze_ticker(ticker):
    print(f"\n▶ Analyzing {ticker}...")

    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    name = info.get("shortName") or info.get("longName") or "N/A"

    df = yf.download(ticker, start="2022-01-01", group_by="ticker", auto_adjust=False)

    if df.empty:
        print(f"⚠️ No data for {ticker}")
        return

    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df[ticker].copy()
        except KeyError:
            print(f"⚠️ Could not extract sub-DataFrame for {ticker}")
            return

    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            print(f"⚠️ No 'Close' or 'Adj Close' column in {ticker}")
            return

    df = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df.dropna(subset=["y"], inplace=True)

    today = datetime.today().date()
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=DAYS_FORWARD)
    forecast = model.predict(future)
    forecast["date_only"] = forecast["ds"].dt.date

    today_row = forecast[forecast["date_only"] == today]

    if today_row.empty:
        print(f"⚠️ No forecast available for {today}")
        return

    yhat_lower = today_row['yhat_lower'].values[0]
    yhat_upper = today_row['yhat_upper'].values[0]
    yhat = today_row['yhat'].values[0]

    latest_price = df['y'].iloc[-1]

    if latest_price < yhat_lower:
        alert = "BUY"
    elif latest_price > yhat_upper:
        alert = "SELL"
    else:
        alert = "HOLD"

    # Add to results summary
    results.append({
        "Ticker": ticker,
        "Name": name,
        "Current Price": format_currency(ticker, latest_price),
        "Forecast (yhat)": format_currency(ticker, yhat),
        "Lower Bound": format_currency(ticker, yhat_lower),
        "Upper Bound": format_currency(ticker, yhat_upper),
        "Alert": alert
    })

def write_markdown(df):
    output_dir = Path("docs")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = output_dir / f"forecast_{timestamp}.md"

    with open(filename, "w") as f:
        f.write(f"# Forecast Report ({timestamp})\n\n")
        f.write(df.to_markdown(index=False))

    print(f"\n✅ Markdown report written to: {filename}")

    # Append to docs/index.md
    index_path = output_dir / "index.md"
    index_path.touch(exist_ok=True)

    with open(index_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    with open(index_path, "w") as index_file:
        index_file.write(f"- [{timestamp}](./forecast_{timestamp}.md)\n")
        index_file.writelines(lines)

    print(f"🔗 Index updated: {index_path}")

if __name__ == "__main__":
    for ticker in TICKERS:
        analyze_ticker(ticker)

    # Display summary table
    df_results = pd.DataFrame(results)
    print("\n📋 Forecast Summary Table:\n")
    print(df_results.to_string(index=False))

    write_markdown(df_results)

    # Optionally export to CSV
    # df_results.to_csv("forecast_summary.csv", index=False)

    print("\n✅ Forecast analysis complete.")