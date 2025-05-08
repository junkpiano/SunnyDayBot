#!/usr/bin/env python
"""
Forecast NASDAQ, S&P500, JEPQ, JEPI with Prophet
and label each asset BUY / HOLD / SELL
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Silence fbprophet msgs

import yfinance as yf
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

from curl_cffi import requests

# ---------- configuration ----------
TICKERS = {
    "^IXIC": "NASDAQ",
    "^GSPC": "S&P500",
    "^N225": "Nikkei225",
}
HISTORY_PERIOD = "5y"   # look-back window for training
FCAST_DAYS     = 30     # forecast horizon
THRESH         = 0.02   # ±2 % band for HOLD
# -----------------------------------

def download_price_series(ticker: str) -> pd.DataFrame:
    """Fetch daily closes, return Prophet-ready DataFrame (columns: ds, y)."""
    session = requests.Session(impersonate="chrome")
    data = yf.download(ticker, period=HISTORY_PERIOD, auto_adjust=False, progress=False, session=session)

    # Ensure we end up with a clean DataFrame [ds, y]
    if data.empty or "Close" not in data.columns:
        return pd.DataFrame()                  # invalid → caller will skip

    data = data.reset_index()[["Date", "Close"]]
    data.columns = ["ds", "y"]
    data["ds"] = pd.to_datetime(data["ds"], errors="coerce")
    data["y"]  = pd.to_numeric(data["y"],  errors="coerce")
    data = data.dropna(subset=["ds", "y"])    # final clean-up
    return data

def prophet_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Fit Prophet and return forecast DataFrame."""
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=FCAST_DAYS)
    return model.predict(future), model

def classify(actual: float, median: float) -> str:
    """BUY / HOLD / SELL decision."""
    if actual <  (1 - THRESH) * median:
        return "BUY"
    if actual >  (1 + THRESH) * median:
        return "SELL"
    return "HOLD"

def main() -> None:
    results = []

    for tic, name in TICKERS.items():
        print(f"▶ Processing {name} …")
        df = download_price_series(tic)
        if df.empty:
            print(f"   ⚠ No valid data, skipped.")
            continue

        forecast, model = prophet_forecast(df)

        # today’s close = last row of original df
        actual_price = df.iloc[-1]["y"]

        # use last row of forecast horizon for comparison
        last_row = forecast.tail(1).iloc[0]
        median   = last_row["yhat"]
        upper    = last_row["yhat_upper"]
        lower    = last_row["yhat_lower"]

        action   = classify(actual_price, median)

        results.append(
            dict(
                Asset=name,
                ActualPrice=round(actual_price, 2),
                ForecastMedian=round(median, 2),
                ForecastUpper=round(upper, 2),
                ForecastLower=round(lower, 2),
                Action=action,
            )
        )

        # save chart
        fig = model.plot(forecast)
        plt.title(name)
        plt.tight_layout()
        fig.savefig(f"./docs/{name}_forecast.png")
        plt.close(fig)

    if results:
        md = pd.DataFrame(results).to_markdown(index=False)
        print("\n" + md + "\n")

        # Write the markdown table and images to ./docs/index.md
        with open("./docs/index.md", "w") as f:
            f.write("# Forecast Summary\n\n")
            f.write(md + "\n\n")
            f.write("## Forecast Charts\n\n")
            for result in results:
                asset_name = result["Asset"]
                image_path = f"{asset_name}_forecast.png"
                f.write(f"### {asset_name}\n\n")
                f.write(f"![{asset_name} Forecast](./{image_path})\n\n")
    else:
        print("⚠ No forecasts were produced (all data unavailable).")

if __name__ == "__main__":
    main()
