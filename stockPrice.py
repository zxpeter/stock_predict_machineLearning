import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

start = dt.datetime(2000, 1, 1)
end = dt.datetime(2017, 12, 1)

# df = web.DataReader('601318.SS', 'yahoo', start, end)  # Ping An Insurance 601318
# print(df.head())  # default first 5
# print(df.head(6))
# print(df.tail(6))
# df.to_csv('601318_data.csv')

df = pd.read_csv('601288_data.csv', parse_dates=True, index_col=0)  # 后面两个参数负责输出以date为index的数组
print(df.head())  # head means first 5 data
# print(df['Date'])

# print(df[['Open', 'High']].head())
# df['Adj Close'].plot()
# plt.show()
#
# df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()
# df.dropna(inplace=True)

df_ohlc = df['Adj Close'].resample('10D').ohlc()  # 重新选取10天为采样周期openhighlowclose
df_volume = df['Volume'].resample('10D').sum()  # 求10天的和
df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
print(df_ohlc.head())

ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()

# print(df_ohlc.head())
candlestick_ohlc(ax1, df_ohlc.values, width=2, colordown='g', colorup='r')  # candle plot
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
# df['Volume'] = df['Volume'].astype(float)
# print(df['Volume'])
# ax1.plot(df.index, df['Adj Close'])
# ax1.plot(df.index, df['100ma'])
# ax2.bar(df.index, df['Volume'])  # bar !!

plt.show()
