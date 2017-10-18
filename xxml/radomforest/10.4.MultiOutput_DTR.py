#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
#多输出的决策树回归，有的时候，输出不一定只是一个。输出结果 标记之间会有些关系的。这个可以看word 记录
if __name__ == "__main__":
    #300个样本 ，每个样本从   x 是从-4 到4 
    N = 300
    x = np.random.rand(N) * 8 - 4     # [-4,4)
    x.sort()
    y1 = np.sin(x) + 3 + np.random.randn(N) * 0.1
    y2 = np.cos(0.3*x) + np.random.randn(N) * 0.01
    # y1 = np.sin(x) + np.random.randn(N) * 0.05
    # y2 = np.cos(x) + np.random.randn(N) * 0.1
    #y1 是一行 y2 是一行   vstck 一下 y 就是2行  叠加在一起
    y = np.vstack((y1, y2))
    #专置一下就得到了y 的  2 列数据
    y = np.vstack((y1, y2)).T
    x = x.reshape(-1, 1)  # 转置后，得到N个样本，每个样本都是1维的

    deep = 3
    #决策回归  深度3   标准是均方误差
    reg = DecisionTreeRegressor(criterion='mse', max_depth=deep)
    #x 是一列 数据    y是2列   是多输出的  回归树，去做拟合
    dt = reg.fit(x, y)

    #做好拟合  去做预测
    x_test = np.linspace(-4, 4, num=1000).reshape(-1, 1)
    print x_test
    y_hat = dt.predict(x_test)
    print y_hat
    plt.scatter(y[:, 0], y[:, 1], c='r', s=40, label='Actual')
    plt.scatter(y_hat[:, 0], y_hat[:, 1], c='g', marker='s', s=100, label='Depth=%d' % deep, alpha=1)
    plt.legend(loc='upper left')
    plt.xlabel('y1')
    plt.ylabel('y2')
    plt.grid()
    plt.show()
