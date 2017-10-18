#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


if __name__ == "__main__":
    #还是用的鸢尾花数据
    data = np.loadtxt('../regression/8.iris.data', dtype=float, delimiter=',', converters={4: iris_type})
    print data
    #前4列是x  最后一列是y
    x, y = np.split(data, (4,), axis=1)
    #取前2个特征
    x = x[:, :2]
    print x
    print y

#这里假定任何一个类别都是高斯分布，假定第一个类被花萼 宽度和高度 服从一个均值和方差的高斯分布，第二个类别服从另外一个均值和方差的高斯分布，并且认为特征之间是独立的
#这样的假定如果特征数目不是特别多的时候，一般没有必要来这么做
#可以用StandardScaler  预处理，把每一个特征都变成 均值是0  方差是1 的标准，救是做标准化的拉升
    gnb = Pipeline([
        ('sc', StandardScaler()),
        ('clf', GaussianNB())])
    #然后把x 和y 放进去   这个函数要求 是行向量
    gnb.fit(x, y.ravel())
    # gnb = MultinomialNB().fit(x, y.ravel())
    # gnb = KNeighborsClassifier(n_neighbors=5).fit(x, y.ravel())

    # 画图
    N, M = 500, 500     # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()   # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()   # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)                    # 生成网格采样点
    x_test = np.stack((x1.flat, x2.flat), axis=1)   # 测试点

    # 无意义，只是为了凑另外两个维度
    # x3 = np.ones(x1.size) * np.average(x[:, 2])
    # x4 = np.ones(x1.size) * np.average(x[:, 3])
    # x_test = np.stack((x1.flat, x2.flat, x3, x4), axis=1)  # 测试点

#这块保证中文
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    #给定3个浅色 3个深色
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_hat = gnb.predict(x_test)                  # 预测值
    y_hat = y_hat.reshape(x1.shape)                 # 使之与输入的形状相同
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_hat, cmap=cm_light)     # 预测值的显示
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)    # 样本的显示
    plt.xlabel(u'花萼长度', fontsize=14)
    plt.ylabel(u'花萼宽度', fontsize=14)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'GaussianNB对鸢尾花数据的分类结果', fontsize=18)
    plt.grid(True)
    plt.show()

    # 训练集上的预测结果
    y_hat = gnb.predict(x)
    y = y.reshape(-1)
    result = y_hat == y
    print y_hat
    print result
    acc = np.mean(result)
    print '准确度: %.2f%%' % (100 * acc)
