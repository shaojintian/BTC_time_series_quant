import requests
from datetime import date,datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests, zipfile, io

## 当前交易对
Info = requests.get('https://fapi.binance.com/fapi/v1/exchangeInfo')
symbols = [s['symbol'] for s in Info.json()['symbols']]
symbols = list(filter(lambda x: x[-4:] == 'USDT', [s.split('_')[0] for s in symbols]))
print(symbols)

#获取任意周期K线的函数
def GetKlines(symbol='BTCUSDT',start='2020-8-10',end='2021-8-10',period='1h',base='fapi',v = 'v1'):
    Klines = []
    start_time = int(time.mktime(datetime.strptime(start, "%Y-%m-%d").timetuple()))*1000 + 8*60*60*1000
    end_time =  min(int(time.mktime(datetime.strptime(end, "%Y-%m-%d").timetuple()))*1000 + 8*60*60*1000,time.time()*1000)
    intervel_map = {'m':60*1000,'h':60*60*1000,'d':24*60*60*1000}
    while start_time < end_time:
        mid_time = start_time+1000*int(period[:-1])*intervel_map[period[-1]]
        url = 'https://'+base+'.binance.com/'+base+'/'+v+'/klines?symbol=%s&interval=%s&startTime=%s&endTime=%s&limit=1000'%(symbol,period,start_time,mid_time)
        res = requests.get(url)
        res_list = res.json()
        if type(res_list) == list and len(res_list) > 0:
            start_time = res_list[-1][0]+int(period[:-1])*intervel_map[period[-1]]
            Klines += res_list
        if type(res_list) == list and len(res_list) == 0:
            start_time = start_time+1000*int(period[:-1])*intervel_map[period[-1]]
        if mid_time >= end_time:
            break
    df = pd.DataFrame(Klines,columns=['time','open','high','low','close','amount','end_time','volume','count','buy_amount','buy_volume','null']).astype('float')
    df.index = pd.to_datetime(df.time,unit='ms')
    return df

start_date = '2022-1-1'
end_date = '2022-09-14'
period = '1h'
df_dict = {}
for symbol in symbols:
    df_s = GetKlines(symbol=symbol,start=start_date,end=end_date,period=period,base='fapi',v='v1')
    if not df_s.empty:
        df_dict[symbol] = df_s

print(df_dict.keys())



