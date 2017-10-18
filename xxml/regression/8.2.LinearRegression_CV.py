#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
#加如l2 正则画  岭回归   l1 的正则画是lasso
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    # pandas读入
    data = pd.read_csv('8.Advertising.csv')    # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data['Sales']
    print x
    print y

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    # print x_train, y_train
    #用l1 正则画
#     model = Lasso()
    #用l2正则画
    model = Ridge()
    #取一个等比数列  最小值 是10  -3 次方  最大是10 的平方    分成10分就是得到10个数  这10个数 是等比数列，这里是等到一些候选的 λ
    alpha_can = np.logspace(-3, 2, 10)
    #加上正则画 需要输入一些超参数  超参数可能是一个alpha 值    这个值需要人工指定，可能是0.1  0.2  3  。怎么样使指定的参数更合适，往往使用交叉验证
    #GridSearchCV  专门做交叉验证的
    #把alpha_can 给这个模型    这里取的是5折的交叉验证
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    lasso_model.fit(x, y)
    print '验证参数：\n', lasso_model.best_params_

    y_hat = lasso_model.predict(np.array(x_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print mse, rmse

    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
