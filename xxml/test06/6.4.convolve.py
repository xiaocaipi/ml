# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == "__main__":
    #数据的分割符号 是tab   前2行是不要的 skiprows=2   就把 2 3 4 5 这4列数据 读取出来 是  最高 最低  收盘 和成交量  unpack是false 的话 返回是一个 加在一起的，如果是true的话  返回是一个元组
    stock_max, stock_min, stock_close, stock_amount = np.loadtxt('6.SH600000.txt', delimiter='\t', skiprows=2, usecols=(2, 3, 4, 5), unpack=True)
    #这个数据是很多的，这里就取前面 100 个
    N = 100
    stock_close = stock_close[:N]
    print stock_close

    n = 5
    #做一个全1 长度是5  这里是5日均线
    weight = np.ones(n)
    #去除以和 得到权值
    weight /= weight.sum()
    print weight
    #利用卷集操作 卷积按道理是有一些补0 是否有效操作的，这里就有效了
    stock_sma = np.convolve(stock_close, weight, mode='valid')  # simple moving average

    #从1 到0 有n个值的等差数列
    weight = np.linspace(1, 0, n)
    #等差数列 取一下e 就是等比数列了
    weight = np.exp(weight)
    # 等比数列做一个均一话 
    weight /= weight.sum()
    #这个 卷积 权值打印下 ，就是今天的收盘价的权值要大一点 前一天小一点  越前面 权值越小   这就是所谓的指数的移动平均线
    print weight
    stock_ema = np.convolve(stock_close, weight, mode='valid')  # exponential moving average
    #t 是从第4天开始有数据 
    t = np.arange(n-1, N)
    #去做一个诺干个度的多项式拟合  t是时间轴   stock_ema 是指数的滑动平均   这个指数滑动平均 想要去拟合  最简单用一个10次的函数去做拟合
    poly = np.polyfit(t, stock_ema, 10)
    #这个poly 就是多项式 10次的多项式
    print poly
    #可以试一下 对于这个多项式 poly  给定一个t  输出的预测值是什么-++
    stock_ema_hat = np.polyval(poly, t)

    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(np.arange(N), stock_close, 'ro-', linewidth=2, label=u'原始收盘价')
    t = np.arange(n-1, N)
    plt.plot(t, stock_sma, 'b-', linewidth=2, label=u'简单移动平均线')
    plt.plot(t, stock_ema, 'g-', linewidth=2, label=u'指数移动平均线')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(9, 6))
    plt.plot(np.arange(N), stock_close, 'r-', linewidth=1, label=u'原始收盘价')
    plt.plot(t, stock_ema, 'g-', linewidth=2, label=u'指数移动平均线')
    plt.plot(t, stock_ema_hat, 'm-', linewidth=3, label=u'指数移动平均线估计')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
