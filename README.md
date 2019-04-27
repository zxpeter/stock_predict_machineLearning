# Stock Predictor using machine learning
According to the historical transaction data of the Shanghai and Shenzhen 300 Index, the price of the stock (the probability of rise and fall) is predicted by the machine learning method, and the corresponding trading strategy is formed by this.
In this experiment, AdjClose is selected as the main sample feature. Finally, choose the hybrid SVM classifier to get the best Accuracy=0.7 result.

针对沪深300指数的历史交易数据，通过机器学习的方法，预测股票的价格（涨跌概率），并借此来形成相应的交易策略。
在本文选取AdjClose即经调整后的收盘价格作为主要样本特征。最后选择混合SVM分类器得到最佳Accuracy=0.7的结果。
An adjusted closing price is a stock's closing price on any given day of tradingthat has been amended to include any distributions and corporate actions thatoccurred at any time prior to the next day's open.

#  How to run
You can change dataReader directory and run process_for_ml.py 
