import yfinance as yf
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# データ取得
nasdaq = yf.download("^IXIC", start="2022-01-01", end=None)

# Prophet用形式へ変換
df = nasdaq.reset_index()[['Date', 'Close']]
df.columns = ['ds', 'y']

# Prophetモデル構築・学習
model = Prophet(daily_seasonality=True)
model.fit(df)

# 30日先までの予測
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# 予測と実績の分離
split_date = df['ds'].max()

# プロット
plt.figure(figsize=(14, 6))

# 実績データ
plt.plot(df['ds'], df['y'], label='Actual (Historical)', color='black')

# 予測の中央値（中心値）
plt.plot(forecast['ds'], forecast['yhat'], label='Forecast (yhat)', color='blue', linestyle='--')

# 予測区間（95% 信頼区間）
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'],
                 color='skyblue', alpha=0.3, label='95% Confidence Interval')

# 予測期間に色をつける（視覚的にわかりやすく）
plt.axvspan(split_date, forecast['ds'].max(), color='lightgrey', alpha=0.2)

# 装飾
plt.title("NASDAQ Forecast (Prophet - Highlighted View)")
plt.xlabel("Date")
plt.ylabel("NASDAQ Index Price")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

for symbol in ["HDV", "SCHD", "ACWI", "^IXIC"]:
    data = get_price_data(symbol)
    model = Prophet()
    model.fit(data)
    forecast = model.predict(future)

    if current_price < forecast["yhat_lower"].iloc[-1]:
        print(f"📉 {symbol} is below lower band. Consider buying.")
