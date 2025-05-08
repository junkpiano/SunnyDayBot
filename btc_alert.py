import requests

def get_btc_daily_change(vs_currency: str = "usd") -> float:
    """
    Fetches the last 2 days of closing prices from the CoinGecko market_chart endpoint
    and returns the day-over-day percentage change.
    """
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": 2,           # fetch data for the last 2 days
        "interval": "daily"  # retrieve daily data points
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    prices = data.get("prices", [])
    if len(prices) < 2:
        raise ValueError("Insufficient data retrieved")

    # prices is a list of [timestamp_ms, price] entries
    price_yesterday = prices[0][1]
    price_today     = prices[1][1]

    # calculate percentage change
    change_pct = (price_today - price_yesterday) / price_yesterday * 100
    return change_pct

if __name__ == "__main__":
    try:
        change = get_btc_daily_change(vs_currency="jpy")
        sign = "+" if change >= 0 else ""
        print(f"BTC daily change: {sign}{change:.2f}%")
    except Exception as e:
        print(f"Error: {e}")

