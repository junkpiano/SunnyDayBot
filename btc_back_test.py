# btc_back_test.py

import yfinance as yf
import pandas as pd
from prophet import Prophet

# === Simulation Configuration ===
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
INITIAL_CASH = 10000  # in USD
TICKER = "BTC-USD"
TRADE_UNIT = 0.01  # BTC per trade

# === Fetch historical data ===
data = yf.download(TICKER, start=START_DATE, end=END_DATE, group_by="ticker", auto_adjust=False)

# MultiIndex handling
if isinstance(data.columns, pd.MultiIndex):
    try:
        data = data[TICKER].copy()
    except KeyError:
        raise ValueError(f"Could not extract data for {TICKER}")

# Fallback for 'Close' column
if "Close" not in data.columns:
    if "Adj Close" in data.columns:
        data["Close"] = data["Adj Close"]
    else:
        raise ValueError(f"No 'Close' or 'Adj Close' column for {TICKER}")

# Prepare data for Prophet
data = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
data["y"] = pd.to_numeric(data["y"], errors="coerce")
data.dropna(inplace=True)

# === Initialize simulation state ===
cash = INITIAL_CASH
btc = 0.0
lookback_days = 90
portfolio_log = []

# === Run simulation ===
for current_idx in range(lookback_days, len(data)):
    history = data.iloc[current_idx - lookback_days:current_idx]
    today = data.iloc[current_idx]
    today_date = today["ds"]
    today_price = today["y"]

    # Train Prophet
    model = Prophet(daily_seasonality=True)
    model.fit(history)
    future = model.make_future_dataframe(periods=1)
    forecast = model.predict(future)

    yhat_lower = forecast["yhat_lower"].iloc[-1]
    yhat_upper = forecast["yhat_upper"].iloc[-1]

    action = "HOLD"

    # Buy signal
    if today_price < yhat_lower and cash >= today_price * TRADE_UNIT:
        btc += TRADE_UNIT
        cash -= today_price * TRADE_UNIT
        action = "BUY"

    # Sell signal
    elif today_price > yhat_upper and btc >= TRADE_UNIT:
        btc -= TRADE_UNIT
        cash += today_price * TRADE_UNIT
        action = "SELL"

    portfolio_value = cash + btc * today_price

    portfolio_log.append({
        "date": today_date,
        "price": today_price,
        "cash": cash,
        "btc": btc,
        "portfolio_value": portfolio_value,
        "action": action
    })

    if portfolio_value <= 0:
        print(f"Portfolio depleted on {today_date}. Simulation ends.")
        break

# === Output results ===
df_result = pd.DataFrame(portfolio_log)
df_result.to_csv("btc_forecast_simulation_result.csv", index=False)
print("\nSimulation completed. Results saved to 'btc_forecast_simulation_result.csv'")