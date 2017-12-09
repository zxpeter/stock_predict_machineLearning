from collections import Counter
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
from sklearn.preprocessing import label_binarize
import time
import datetime as dt
from sklearn import svm, cross_validation, neighbors, metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.svm import SVR
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier


# 计算一周内的每天收益率
def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('csi300_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days + 1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
        # 第7(i)天的收盘价格减去第1天的价格，除以第一天的价格，表示收益率

    df.fillna(0, inplace=True)
    print(df)
    return tickers, df  # 形成n+7维数据df


# 决定在当前时间点是否买入/卖出/持有
def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02  # 对于当天价格，预测一周后会涨，收益超过2%
    for col in cols:
        if col > requirement:  # 则买入(上下可以不一样)
            return 1
        if col < -requirement:  # 则卖出 !!注意是负数
            return 2
    return 0


# 构造样本数据 X维度为所有股票的昨天到今天的收益  y为一周之内距离今天最近的一天是涨,买(1)还是跌，卖(-1)
def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    # print(df.head())
    # 综合起来看是 买/卖/持有
    # 预测未来7天，只要离得最近的一天涨/跌，就买/卖，一直是0则不管
    # returns a 1, or -1 if ANY of the 7 inputs exceeds requirement, not a value for each input
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                              df['{}_1d'.format(ticker)],
                                              df['{}_2d'.format(ticker)],
                                              df['{}_3d'.format(ticker)],
                                              df['{}_4d'.format(ticker)],
                                              df['{}_5d'.format(ticker)],
                                              df['{}_6d'.format(ticker)],
                                              df['{}_7d'.format(ticker)]
                                              ))
    vals = df['{}_target'.format(ticker)].values.tolist()  # 每只股票是一个样本，一个样本是一个list
    # print(df.head())
    # print(vals)  # [0, -1, 1]
    str_vals = [str(i) for i in vals]
    # print(str_vals)  # ['0', '-1', '1']
    print('data spread:', Counter(str_vals))  # 计算所有 0 -1 1 标签的个数

    # 整理数据
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    # 正则化
    # df_vals = df[[ticker]].pct_change()  # 用来计算同colnums两个相邻的数字之间的变化率,percent change yesterday
    # df_vals = df[[ticker]]  # 可以只看自己的价格，也可以看整个大盘，不同股票的价格之间有相关性
    # df_vals = df[[ticker for ticker in tickers]]  # 这种方式比上面准确率大提升
    df_vals = df[[ticker for ticker in tickers]].pct_change()  # 这种方式比上面准确率大提升
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    df.reset_index(inplace=True)  # 重设index为序列，将date作为一个column

    X = df_vals.values  # 4271 * 1(Adj Close) * n(tickers)
    y = df['{}_target'.format(ticker)].values

    return X, y, df


# extract_featuresets('MMM')

def do_machineLearning(ticker):
    X, y, df = extract_featuresets(ticker)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)  # 切分
    # clf = VotingClassifier([('lsvc', svm.LinearSVC()),
    #                         ('knn', neighbors.KNeighborsClassifier()),
    #                         ('rfor', RandomForestClassifier())])
    # clf = MLPClassifier()
    # clf = neighbors.KNeighborsClassifier()
    # clf = svm.LinearSVC()
    clf = RandomForestClassifier()
    # print(X_test.shape)
    # print(X_test.shape)
    clf.fit(X_train, y_train)

    random_state = np.random.RandomState(0)
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                             random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_test_test = label_binarize(y_test, classes=[0, 1, 2])
    # print(y_test_test)
    for i in range(3):  # 3分类问题
        fpr[i], tpr[i], _ = roc_curve(y_test_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # print(y_test_test[:, 1])

    roc_auc_score = metrics.roc_auc_score(y_test_test, y_score, average='macro')
    print('roc_auc_score:', roc_auc_score)

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    lw = 2
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= 3
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    confidence = clf.score(X_test, y_test)
    print('Accuracy:', confidence)
    prediction = clf.predict(X_test)
    print('Predicted spread:', Counter(prediction))
    precision = metrics.precision_score(y_test, prediction, average='weighted')
    print('precision:', precision)
    recall_score = metrics.recall_score(y_test, prediction, average='weighted')
    print('recall_score:', recall_score)
    f1_score = metrics.f1_score(y_test, prediction, average='weighted')
    print('f1_score:', f1_score)


    target_names = ['class-1', 'class0', 'class1']
    print(metrics.classification_report(y_test, prediction, target_names=target_names))

    print(X_test)
    print(prediction)
    # ppp2 = clf.predict(100)  # when price equal to 100
    # print(ppp2)

    return confidence


# 你要预测的是某一天当这只股票的价格为X的时候，是涨还是跌，是买还是卖

do_machineLearning('601318.SS')  # Ping An Insurance 601318


# 以x=时间,y=价格为训练样本，主要判断模型对数据的拟合度
def extract_yy(ticker):
    tickers, df = process_data_for_labels(ticker)
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df_vals = df[[ticker]]  # 可以只看自己的价格，也可以看整个大盘，不同股票的价格之间有相关性
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)
    df.reset_index(inplace=True)  # 重设index为序列，将date作为一个column
    df['Date'] = [time.mktime(dt.datetime.strptime(i, "%Y-%m-%d").timetuple()) for i in df['Date']]
    yy = df[['Date']]
    yy = yy.values
    X = df_vals.values  # 4271 * 1(Adj Close) * n(tickers)
    return X, yy


def do_ml_plot(ticker):
    prices, dates = extract_yy(ticker)
    clf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    clf.fit(dates, prices)
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, clf.predict(dates), color='red', label='Model')
    plt.show()
    prediction = clf.predict(dates)
    print('Predicted price:', prediction)

    return prediction

# do_ml_plot('MMM')
