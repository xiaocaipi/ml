# !/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'


def expand(a, b, rate=0.05):
    d = (b - a) * rate
    return a-d, b+d


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


if __name__ == '__main__':
    path = '../regression/8.iris.data'  # 数据文件路径
    #还是用iris_type 回调函数把 y变成 012
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    # 将数据的0到3列组成x，第4列得到y
    x_prime, y = np.split(data, (4,), axis=1)
    y = y.ravel()

    n_components = 3
    #枚举特征  0 1 2列  02  2列
    feature_pairs = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    plt.figure(figsize=(10, 9), facecolor='#FFFFFF')
    for k, pair in enumerate(feature_pairs):
        x = x_prime[:, pair]
        #y 是实际的标记值  把y 是0 1 2 做遍历 比如 i=1 的时候  就把y=1 的值拿出来  x特征值拿出来之后 直接求均值 就得到第一个类别的均值   也可以做第二个  第三个
        m = np.array([np.mean(x[y == i], axis=0) for i in range(3)])  # 均值的实际值
        print '实际均值 = \n', m
        #分成3个类别  这3个类别的方差 是没有必要相同的
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
        #做拟合 ，把均值 和方差算出来
        gmm.fit(x)
        print '预测均值 = \n', gmm.means_
        print '预测方差 = \n', gmm.covariances_
        #把x带进去 就能算y_hat
        y_hat = gmm.predict(x)
        #m 是实际的 均值  gmm.means_  是预测的3个类别均值，其实顺序可能是不一样的
        order = pairwise_distances_argmin(m, gmm.means_, axis=1, metric='euclidean')
        #这里打印出来是 0 1 2  就非常漂亮  但打出来是 0 2 1   实际是第0个类被 预测出来是第0个类别  实际第1个类别预测出第2个类别    实际第2个 预测出 第1个、
        #因此要利用这个order 做一个转换
        # print '顺序：\t', order

        #做转换  y是 样本个数
        n_sample = y.size
        n_types = 3
        #造 n_types   n_sample  布尔的数组
        change = np.empty((n_types, n_sample), dtype=np.bool)
       
        for i in range(n_types):
             #i=0 的时候 把order 0 拿出来 就是0 把y_hat =0 的拿出来 这些值应该是0
             #如果i=1  order 1  是等于2   把yy_hat 是2的值拿出来 ，这些预测是2 的实际应该是1   就是把类别该一下顺序，分的名字顺序一下
             #i=2  order 2 是1   yy_hat 是1 拿出来  这些实际是2
             #这样 y_hat 顺序也是 0 1 2
            change[i] = y_hat == order[i]
        for i in range(n_types):
            y_hat[change[i]] = i
        #这样顺序一样了 才能 把y_hat  和y 去做 准确率
        acc = u'准确率：%.2f%%' % (100*np.mean(y_hat == y))
        print acc

        cm_light = mpl.colors.ListedColormap(['#FF8080', '#77E0A0', '#A0A0FF'])
        cm_dark = mpl.colors.ListedColormap(['r', 'g', '#6060FF'])
        x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
        x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
        x1_min, x1_max = expand(x1_min, x1_max)
        x2_min, x2_max = expand(x2_min, x2_max)
        x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]
        grid_test = np.stack((x1.flat, x2.flat), axis=1)
        grid_hat = gmm.predict(grid_test)

        change = np.empty((n_types, grid_hat.size), dtype=np.bool)
        for i in range(n_types):
            change[i] = grid_hat == order[i]
        for i in range(n_types):
            grid_hat[change[i]] = i

        grid_hat = grid_hat.reshape(x1.shape)
        plt.subplot(3, 2, k+1)
        plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
        plt.scatter(x[:, 0], x[:, 1], s=30, c=y, marker='o', cmap=cm_dark, edgecolors='k')
        xx = 0.95 * x1_min + 0.05 * x1_max
        yy = 0.1 * x2_min + 0.9 * x2_max
        plt.text(xx, yy, acc, fontsize=14)
        plt.xlim((x1_min, x1_max))
        plt.ylim((x2_min, x2_max))
        plt.xlabel(iris_feature[pair[0]], fontsize=14)
        plt.ylabel(iris_feature[pair[1]], fontsize=14)
        plt.grid()
    plt.tight_layout(2)
    plt.suptitle(u'EM算法无监督分类鸢尾花数据', fontsize=20)
    plt.subplots_adjust(top=0.92)
    plt.show()
