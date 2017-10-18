# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin


mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False


if __name__ == '__main__':
    style = 'sklearn'
    #首先给一个随机的种子值  保证所有样本是一致
    np.random.seed(0)
    #然后给定一个 u1  和方差     第一个类别的均值是原点，这里是3个唯独
    mu1_fact = (0, 0, 0)
    #方差是 单位阵
    cov_fact = np.identity(3)
    #用了400个正例  100个负例  用多元正太分布  只要给定均值 和方差  400  是返回400 个样本
    data1 = np.random.multivariate_normal(mu1_fact, cov_fact, 400)
    # 同样是u2  
    mu2_fact = (2, 2, 1)
    #还是单位阵
    cov_fact = np.identity(3)
    data2 = np.random.multivariate_normal(mu2_fact, cov_fact, 100)
    #把data1 和2  用垂直方向做叠加  那么这个data  是500 个 前400个是true  后100 是false
    data = np.vstack((data1, data2))
    #给类别  y ，这个类别是不用的
    y = np.array([True] * 400 + [False] * 100)

    if style == 'sklearn':
        # 给出2个类别，full  表明方差的类型 默认是full  有圆型的 椭圆的  tired 的   tol 是容差，就是多少以下就不再迭代了   max_iter  最大迭代次数
        g = GaussianMixture(n_components=2, covariance_type='full', tol=1e-6, max_iter=1000)
        g.fit(data)
        # 类别概率 就是  第一个类别占多少
        print '类别概率:\t', g.weights_[0]
        #2个类别的均值
        print '均值:\n', g.means_, '\n'
        #2个类别的方差  实际是2个单位阵，但这里不是
        print '方差:\n', g.covariances_, '\n'
        mu1, mu2 = g.means_
        sigma1, sigma2 = g.covariances_
    else:
        num_iter = 100
        #有n行 d列  n就是样本个数  d就是维度
        n, d = data.shape
        # 随机指定
        # mu1 = np.random.standard_normal(d)
        # print mu1
        # mu2 = np.random.standard_normal(d)
        # print mu2
        # 样本最小值 u1  最大值是u2  这样就能把样本去分开
        mu1 = data.min(axis=0)
        mu2 = data.max(axis=0)
        sigma1 = np.identity(d)
        sigma2 = np.identity(d)
        pi = 0.5
        # EM
        for i in range(num_iter):
            # E Step
            norm1 = multivariate_normal(mu1, sigma1)
            norm2 = multivariate_normal(mu2, sigma2)
            tau1 = pi * norm1.pdf(data)
            tau2 = (1 - pi) * norm2.pdf(data)
            gamma = tau1 / (tau1 + tau2)

            # M Step
            mu1 = np.dot(gamma, data) / np.sum(gamma)
            mu2 = np.dot((1 - gamma), data) / np.sum((1 - gamma))
            sigma1 = np.dot(gamma * (data - mu1).T, data - mu1) / np.sum(gamma)
            sigma2 = np.dot((1 - gamma) * (data - mu2).T, data - mu2) / np.sum(1 - gamma)
            pi = np.sum(gamma) / n
            print i, ":\t", mu1, mu2
        print '类别概率:\t', pi
        print '均值:\t', mu1, mu2
        print '方差:\n', sigma1, '\n\n', sigma2, '\n'

    # 预测分类  这里均值和方差 有了 ，那就去用这2个 造 2个正太分布
    norm1 = multivariate_normal(mu1, sigma1)
    norm2 = multivariate_normal(mu2, sigma2)
    #tau1  就是norm1 高斯分布下  概率密度函数的值   如果tau1 > tau2 就是一个类别  否则就是另外一个类别
    tau1 = norm1.pdf(data)
    tau2 = norm2.pdf(data)

    fig = plt.figure(figsize=(13, 7), facecolor='w')
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', s=30, marker='o', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(u'原始数据', fontsize=18)
    ax = fig.add_subplot(122, projection='3d')
    #比对一下 算得mu1  和mu1_fact  真是的mu1 来比 是不是   返回的order  是01  或者是10
    order = pairwise_distances_argmin([mu1_fact, mu2_fact], [mu1, mu2], metric='euclidean')
    #如果是0  
    if order[0] == 0:
        c1 = tau1 > tau2
    else:
        c1 = tau1 < tau2
    #c1 的反就是c2
    c2 = ~c1
    acc = np.mean(y == c1)
    print u'准确率：%.2f%%' % (100*acc)
    ax.scatter(data[c1, 0], data[c1, 1], data[c1, 2], c='r', s=30, marker='o', depthshade=True)
    ax.scatter(data[c2, 0], data[c2, 1], data[c2, 2], c='g', s=30, marker='^', depthshade=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(u'EM算法分类', fontsize=18)
    # plt.suptitle(u'EM算法的实现', fontsize=20)
    # plt.subplots_adjust(top=0.92)
    plt.tight_layout()
    plt.show()
