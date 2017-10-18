# /usr/bin/python
# -*- encoding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import csv

#这个文档里面有解释的
def f(x):
    return 0.5*np.exp(-(x+3) **2) + np.exp(-x**2) + + 0.5*np.exp(-(x-3) ** 2)


if __name__ == "__main__":
    #每次运行得到相同的随机数
    np.random.seed(0)
    N = 200
    x = np.random.rand(N) * 10 - 5  # [-5,5)
    x = np.sort(x)
    #f(x) 就是造的函数，
    y = f(x) + 0.05*np.random.randn(N)
    x.shape = -1, 1
    #用岭回归
    ridge = RidgeCV(alphas=np.logspace(-3, 2, 10), fit_intercept=False)
    #叠加了一个 度，就是x 是几次方
    ridged = Pipeline([('poly', PolynomialFeatures(degree=10)), ('Ridge', ridge)])
    #最后放入 bagging里面，用30% 数据做重采样，做100次
    bagging_ridged = BaggingRegressor(ridged, n_estimators=100, max_samples=0.3)
    #决策树
    dtr = DecisionTreeRegressor(max_depth=5)
    regs = [
        ('DecisionTree Regressor', dtr),
        ('Ridge Regressor(6 Degree)', ridged),
        ('Bagging Ridge(6 Degree)', bagging_ridged),
        ('Bagging DecisionTree Regressor', BaggingRegressor(dtr, n_estimators=100, max_samples=0.3))]
    x_test = np.linspace(1.1*x.min(), 1.1*x.max(), 1000)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 8), facecolor='w')
    plt.plot(x, y, 'ro', label=u'训练数据')
    plt.plot(x_test, f(x_test), color='k', lw=3.5, label=u'真实值')
    clrs = 'bmyg'
    for i, (name, reg) in enumerate(regs):
        reg.fit(x, y)
        y_test = reg.predict(x_test.reshape(-1, 1))
        plt.plot(x_test, y_test.ravel(), color=clrs[i], lw=i+1, label=name, zorder=6-i)
    plt.legend(loc='upper left')
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.title(u'回归曲线拟合', fontsize=21)
    plt.ylim((-0.2, 1.2))
    plt.tight_layout(2)
    plt.grid(True)
    plt.show()
