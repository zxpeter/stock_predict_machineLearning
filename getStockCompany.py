import bs4 as bs
import pickle
import requests
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import os
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')


# 获取股票代码
def save_csi300_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/CSI_300_Index')

    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        if row.findAll('td')[2].text == 'Shanghai':
            ticker = ticker + '.SS'
        else:
            ticker = ticker + '.SZ'
        tickers.append(ticker)

    with open("csi300tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    print(tickers)

    return tickers


# save_csi300_tickers()


# 从Yahoo finance获取价格等数据
def get_data_from_yahoo(reload_csi300=False):
    if reload_csi300:
        tickers = save_csi300_tickers()
    else:
        with open("csi300tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_CSI'):
        os.makedirs('stock_CSI')

    start = dt.datetime(2010, 1, 1)
    end = dt.datetime(2017, 12, 1)

    for ticker in tickers[:51]:  # 只获取csi300的前25支ticker
        print(ticker)
        if not os.path.exists('stock_CSI/{}.csv'.format(ticker)):
            if ticker == '300104.SZ':  # leshi 的数据不能获取到
                print('Leshi')
                continue
            else:
                df = web.DataReader(ticker, 'yahoo', start, end)

                df.to_csv('stock_CSI/{}.csv'.format(ticker))
        else:
            print('already have {}'.format(ticker))


# get_data_from_yahoo()


# 将所有股票adjclose数据组合成csv
def compile_data():
    with open("csi300tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers[:51]):  # 只取前25项
        if ticker == '300104.SZ':  # leshi 的数据不能获取到
            print('Leshi')
            continue
        else:
            df = pd.read_csv('stock_CSI/{}.csv'.format(ticker))
            df.set_index('Date', inplace=True)

            df.rename(columns={'Adj Close': ticker}, inplace=True)
            df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')

            print(count)

    print(main_df.head())
    main_df.to_csv('csi300_joined_closes.csv')


# compile_data()


# 表征股票价格之间的相关关系，独立性，与大盘的协同性
def visualize_data():
    df = pd.read_csv('csi300_joined_closes.csv')
    # df['MMM'].plot()
    # plt.show()
    df_corr = df.corr()  # 相关性，相关关系，某只股票的调整后的收盘价格和其余股票的价格的相关系数（暂时这样理解
    print(df_corr.head())

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)

    plt.xticks(rotation=90)
    heatmap.set_clim(-1, 1)

    plt.tight_layout()
    plt.show()


visualize_data()
