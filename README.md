# zxpeter-stock_predict_machineLearning
针对沪深300指数的历史交易数据，通过机器学习的方法，预测股票的价格（涨跌概率），并借此来形成相应的交易策略。
在本文选取AdjClose即经调整后的收盘价格作为主要样本特征。最后选择混合SVM分类器得到最佳Accuracy=0.7的结果。

#  how to run
just change dataReader directory and simply run process_for_ml.python 
