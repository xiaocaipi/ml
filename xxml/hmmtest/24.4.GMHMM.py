# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics.pairwise import pairwise_distances_argmin
import warnings


def expand(a, b):
    d = (b - a) * 0.05
    return a-d, b+d


if __name__ == "__main__":
    #hmmlearn  的版本是在0.2.0 左右  而这个里面 是调用了 sklearn 0.17 之前的一些  而自己用的sklearn 是0.18 的，所以会给 warning
    warnings.filterwarnings("ignore")   # hmmlearn(0.2.0) < sklearn(0.18)
    np.random.seed(0)

    #一共选了5个高斯分布
    n = 5   # 隐状态数目
    #1000个样本
    n_samples = 1000
    #做一个长度是5的随机变量
    pi = np.random.rand(n)
    # 对这个随机变量 做归一化，这个就是初始概率
    pi /= pi.sum()
    print '初始概率：', pi
    # 做一个5 *5 的A，
    A = np.random.rand(n, n)
    # 第0 行 1列  0 行4列  清0   去掉这些为0  的作用 是让 相邻的2个点 不能够相连
    mask = np.zeros((n, n), dtype=np.bool)
    mask[0][1] = mask[0][4] = True
    mask[1][0] = mask[1][2] = True
    mask[2][1] = mask[2][3] = True
    mask[3][2] = mask[3][4] = True
    mask[4][0] = mask[4][3] = True
    A[mask] = 0
    #A 矩阵 其它数据  每一行进行归一化，得到A
    for i in range(n):
        A[i] /= A[i].sum()
    print '转移概率：\n', A
    # 然后取5个中心
    means = np.array(((30, 30), (0, 50), (-25, 30), (-15, 0), (15, 0)))
    print '均值：\n', means
    # 再去选5个 2*2 的方差
    covars = np.empty((n, 2, 2))
    for i in range(n):
        # covars[i] = np.diag(np.random.randint(1, 5, size=2))
        #这个方差 取的是对角阵，为了防止为0  加个0.001
        covars[i] = np.diag(np.random.rand(2)+0.001)*10    # np.random.rand ∈[0,1)
    print '方差：\n', covars

    #可以利用  hmm learn    n_components  隐变量 是5个  ，协防差  是full   这个在em 算法里面 说过的 有 tired full spherical
    model = hmm.GaussianHMM(n_components=n, covariance_type='full')
    #把 pi  A  均值 协防差 给到 模型里面去
    model.startprob_ = pi
    model.transmat_ = A
    model.means_ = means
    model.covars_ = covars
    #用 模型去做采样
    sample, labels = model.sample(n_samples=n_samples, random_state=0)

    # 估计参数
    #这里 迭代10 次
    model = hmm.GaussianHMM(n_components=n, covariance_type='full', n_iter=10)
    model = model.fit(sample)
    #对sample 进行估计
    y = model.predict(sample)
    #不要显示科学计数
    np.set_printoptions(suppress=True)
    #就能把估计的4个值给 拿出来
    print '##估计初始概率：', model.startprob_
    print '##估计转移概率：\n', model.transmat_
    print '##估计均值：\n', model.means_
    print '##估计方差：\n', model.covars_

    # 类别  因为实际5个类被的均值  和 估计出来的5个类别的均值的顺序 对应不熵，所以 就那欧拉相似度去计算 顺序
    order = pairwise_distances_argmin(means, model.means_, metric='euclidean')
    print order
    # 对应顺序 的  A  均值等都改一下
    pi_hat = model.startprob_[order]
    A_hat = model.transmat_[order]
    A_hat = A_hat[:, order]
    means_hat = model.means_[order]
    covars_hat = model.covars_[order]
    change = np.empty((n, n_samples), dtype=np.bool)
    for i in range(n):
        change[i] = y == order[i]
    for i in range(n):
        y[change[i]] = i
    print '估计初始概率：', pi_hat
    print '估计转移概率：\n', A_hat
    print '估计均值：\n', means_hat
    print '估计方差：\n', covars_hat
    print labels
    print y
    acc = np.mean(labels == y) * 100
    print '准确率：%.2f%%' % acc
    #把数据 显示出来
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.scatter(sample[:, 0], sample[:, 1], s=50, c=labels, cmap=plt.cm.Spectral, marker='o',
                label=u'观测值', linewidths=0.5, zorder=20)
    plt.plot(sample[:, 0], sample[:, 1], 'r-', zorder=10)
    plt.scatter(means[:, 0], means[:, 1], s=100, c=np.random.rand(n), marker='D', label=u'中心', alpha=0.8, zorder=30)
    x1_min, x1_max = sample[:, 0].min(), sample[:, 0].max()
    x2_min, x2_max = sample[:, 1].min(), sample[:, 1].max()
    x1_min, x1_max = expand(x1_min, x1_max)
    x2_min, x2_max = expand(x2_min, x2_max)
    plt.xlim((x1_min, x1_max))
    plt.ylim((x2_min, x2_max))
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
