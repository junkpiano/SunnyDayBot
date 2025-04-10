# alert_if_out_of_bounds.py (includes BTC)

import yfinance as yf
from prophet import Prophet
import pandas as pd
from datetime import datetime

# === Configuration ===
TICKERS = ["^IXIC", "HDV", "BTC-USD", "SCHD", "VYM", "AAPL", "TSLA", "NVDA"]  # NASDAQ, HDV ETF, and Bitcoin in USD
DAYS_FORWARD = 30

def analyze_ticker(ticker):
    print(f"\n▶ Analyzing {ticker}...")

    df = yf.download(ticker, start="2022-01-01", group_by="ticker", auto_adjust=False)

    if df.empty:
        print(f"⚠️ No data for {ticker}")
        return

    # Handle MultiIndex column structure
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df[ticker].copy()
        except KeyError:
            print(f"⚠️ Could not extract sub-DataFrame for {ticker}")
            return

    # Check for Close column, fall back to Adj Close if needed
    if "Close" not in df.columns:
        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            print(f"⚠️ No 'Close' or 'Adj Close' column in {ticker}")
            return

    # Prepare data for Prophet
    df = df.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df.dropna(subset=["y"], inplace=True)

    today = datetime.today().date()
    model = Prophet(daily_seasonality=True)
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

    print(f"\n📊 {ticker} Check on {today}")
    print(f"Current Price: {latest_price:.2f}")
    print(f"Forecast: {yhat:.2f} (Range {yhat_lower:.2f} ~ {yhat_upper:.2f})")

    if latest_price < yhat_lower:
        print("📉 Alert: Price is BELOW forecast range. Consider BUYING.")
    elif latest_price > yhat_upper:
        print("📈 Alert: Price is ABOVE forecast range. Consider taking profits.")
    else:
        print("✅ Price is within forecast range. No action needed.")

if __name__ == "__main__":
    for ticker in TICKERS:
        analyze_ticker(ticker)
    print("\n✅ Forecast analysis complete.")
