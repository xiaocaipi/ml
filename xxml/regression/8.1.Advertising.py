#!/usr/bin/python
# -*- coding:utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 #train_test_split 是sklearn  这个库，这个是非常重要的机器学习的库  用这个库下面的model_selection  有一个train_test_split 函数  把xy 随机做采样一部分是训练数据，一部分是测试数据
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    path = '8.Advertising.csv'
    # # 手写读取数据 - 请自行分析，在8.2.Iris代码中给出类似的例子
    # f = file(path)
    # x = []
    # y = []
    #  对f 进行遍历  第0行要跳过的
    # for i, d in enumerate(f):
    #     if i == 0:
    #         continue
    #  d.strip()  相当于 是trim 操作
    #     d = d.strip()
    #     if not d:
    #         continue
    #用，分开  变成float  最后获得一个列表
    #     d = map(float, d.split(','))
    #     x.append(d[1:-1])
    #     y.append(d[-1])
    # print x
    # print y
    # x = np.array(x)
    # y = np.array(y)

    # # Python自带库
    # f = file(path, 'rb')
    # print f
    # d = csv.reader(f)
    # for line in d:
    #     print line
    # f.close()

    # # numpy读入
    # p = np.loadtxt(path, delimiter=',', skiprows=1)
    # print p

    # pandas读入
    data = pd.read_csv(path)    # TV、Radio、Newspaper、Sales
#     x = data[['TV', 'Radio', 'Newspaper']]
    x = data[['TV', 'Radio']]
    y = data['Sales']
    print x
    print y

    # # 绘制1  这个是画在一起 
    plt.plot(data['TV'], y, 'ro', label='TV')
    plt.plot(data['Radio'], y, 'g^', label='Radio')
    plt.plot(data['Newspaper'], y, 'mv', label='Newspaer')
    plt.legend(loc='lower right')
    # plt.grid()
    plt.show()
    # #
    # # 绘制2   这个是分开显示
    plt.figure(figsize=(9,12))
    plt.subplot(311)
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid()
    plt.subplot(312)
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid()
    plt.subplot(313)
    plt.plot(data['Newspaper'], y, 'b*')
    plt.title('Newspaper')
    plt.grid()
    plt.tight_layout()
    plt.show()

    #train_test_split 是sklearn  这个库，这个是非常重要的机器学习的库  train_test_split 把xy 随机做采样一部分是训练数据，一部分是测试数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    # print x_train, y_train
    #建立线性回归模型
    linreg = LinearRegression()
    #利用训练数据进行拟合  返回一个模型
    model = linreg.fit(x_train, y_train)
    print model
    #打印系数
    print linreg.coef_
    #打印截距
    print linreg.intercept_

    #验证模型是否正确，把test数据放进去  得到y的预测值  y_hat 是预测值  y_test是实际值
    y_hat = linreg.predict(np.array(x_test))
    #用平方和的均值 均方误差 
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    #均方误差开根号
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    #这2个值是越小越好
    print mse, rmse

    #把预测值 和实际值 都画出来
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
