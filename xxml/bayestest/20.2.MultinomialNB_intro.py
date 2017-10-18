#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB


if __name__ == "__main__":
    #保证每次随机数据都一样
    np.random.seed(0)
    #做20个样本
    M = 20
    #每个样本都是5维的
    N = 5
    # 这里的随机值 只能取0 或者1   这个就是 m 行n 列的 0 1 矩阵
    x = np.random.randint(2, size=(M, N))     # [low, high)
    # 上面的数据 有可能是重复的，这里做一下去重的工作
    #先对x 做遍历  把这个t 拿出来 做成一个 tuple，  在把这个tuple 放在 set 里面   set 里面的数据是不会重复的  再把set转成一个list
    x = np.array(list(set([tuple(t) for t in x])))
    #M 就是去重了之后的
    M = len(x)
    #认为y  每一行数据 就是一个类别
    y = np.arange(M)
    print '样本个数：%d，特征数目：%d' % x.shape
    print '样本：\n', x
    #做一个多项式 朴素贝叶斯
    mnb = MultinomialNB(alpha=1)    # 动手：换成GaussianNB()试试预测结果？
    #把x 和y 都放进去
    mnb.fit(x, y)
    #直接用x 去预测 y‘
    y_hat = mnb.predict(x)
    print '预测类别：', y_hat
    #手动的去算一下准确率
    print '准确率：%.2f%%' % (100*np.mean(y_hat == y))
    #也可以用score 函数去算一下 
    print '系统得分：', mnb.score(x, y)
    #也可以用sklear 里面的 metrics 来度量准确率的得分
    # from sklearn import metrics
    # print metrics.accuracy_score(y, y_hat)
    err = y_hat != y
    for i, e in enumerate(err):
        if e:
            print y[i], '：\t', x[i], '被认为与', x[y_hat[i]], '一个类别'
