import requests
import pandas as pd
import mplfinance as mpf

# 获取比特币K线数据
url = 'https://api.binance.com/api/v3/klines'
params = {
    'symbol': 'BTCUSDT',
    'interval': '15m',
    'limit': 1000
}
res = requests.get(url, params=params)
data = res.json()
df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

df = df.apply(pd.to_numeric, errors='ignore')

#save
df.to_csv('btc_prices_15min')

# 画K线图
#mpf.plot(df, type='candle', volume=True, style='binance')

print(df.head(10))