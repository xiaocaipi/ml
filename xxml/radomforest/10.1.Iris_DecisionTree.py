#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def iris_type(s):
    it = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return it[s]


# 花萼长度、花萼宽度，花瓣长度，花瓣宽度
# iris_feature = 'sepal length', 'sepal width', 'petal length', 'petal width'
iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == "__main__":
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    path = '../regression/8.iris.data'  # 数据文件路径
    #还是用numpy 把数据读进来  对于第四列数据还是需要一个回跳函数  
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    #data 是所用 数据 前面4列 是x  从第四列开始到最后，也就是最后一列 是y
    x, y = np.split(data, (4,), axis=1)
    # 为了可视化，仅使用前两列特征
    x = x[:, :2]
    #把 xy  测试数据 是30%
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    #ss = StandardScaler()
    #ss = ss.fit(x_train)

    # 决策树参数估计
    # min_samples_split = 10：如果该结点包含的样本数目大于10，则(有可能)对其分支
    # min_samples_leaf = 10：若将某结点分支后，得到的每个子结点样本数目都大于10，则完成分支；否则，不进行分支
    #弄上面这些参数 可以防止树太深 过拟合
    #决策数分类器  最大的深度是3
    model = Pipeline([
        #做了标准化，保证每一个特征 均值都是0  方差都是1  这样往往可以一定程度提高分类效果
        ('ss', StandardScaler()),
        #决策数分类器，用熵的准则 来进行分割，最大的深度是3
        ('DTC', DecisionTreeClassifier(criterion='entropy', max_depth=3))])
    # clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
    #拿到训练数据 可以模型的计算
    model = model.fit(x_train, y_train)
    #对测数数据进行一个预测
    y_test_hat = model.predict(x_test)      # 测试数据

    # 保存
    # dot -Tpng -o 1.png 1.dot  写的方式去生成一个这样的文件
    f = open('.\\iris_tree.dot', 'w')
    #把这个tree 输出到 图的可视化   model 里面的 DTC 就是决策数这个分类器  model.get_params('DTC')['DTC'] 返回的就是一个DecisionTreeClassifier 分类器
    tree.export_graphviz(model.get_params('DTC')['DTC'], out_file=f)
    f.close()
    # 画图
    #横纵坐标  从最小到最大 取100 份
    N, M = 100, 100  # 横纵各采样多少个值
    #x[:, 0]  ： 是所有数据  ，0 是取第0 列，就是取所有数据的第0列  
    #取第0列的最小值  和最大值，同样取 第1列的最小值和最大值。这样就能把图的4个点确定
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    #然后 从最小到最大 做等差数列 100 个
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    #  拼成图中完整的格子的点
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    #把 x1 拉升一维的 x2 也是 拼在一起
    x_show = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    # # 无意义，只是为了凑另外两个维度
    # # 打开该注释前，确保注释掉x = x[:, :2]
    # x3 = np.ones(x1.size) * np.average(x[:, 2])
    # x4 = np.ones(x1.size) * np.average(x[:, 3])
    # x_test = np.stack((x1.flat, x2.flat, x3, x4), axis=1)  # 测试点

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    #model 有了 把 x_show 传进去 来预测一下
    y_show_hat = model.predict(x_show)  # 预测值
    y_show_hat = y_show_hat.reshape(x1.shape)  # 使之与输入的形状相同
    #这里可以做figure  是画边界  用的白色
    plt.figure(facecolor='w')
    #把背景块画出来
    plt.pcolormesh(x1, x2, y_show_hat, cmap=cm_light)  # 预测值的显示
    #把测试数据画上去
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test.ravel(), edgecolors='k', s=100, cmap=cm_dark, marker='o')  # 测试数据
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), edgecolors='k', s=40, cmap=cm_dark)  # 全部数据
    plt.xlabel(iris_feature[0], fontsize=15)
    plt.ylabel(iris_feature[1], fontsize=15)
    # x 就画x最小  到最大  不写这句画，画的时候会留白一部分
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid(True)
    plt.title(u'鸢尾花数据的决策树分类', fontsize=17)
    plt.show()

    # 训练集上的预测结果
    y_test = y_test.reshape(-1)
    print y_test_hat
    print y_test
    result = (y_test_hat == y_test)   # True则预测正确，False则预测错误
    acc = np.mean(result)
    print '准确度: %.2f%%' % (100 * acc)

    # 过拟合：错误率  看一下到底用几层 准确率是最好
    depth = np.arange(1, 15)
    err_list = []
    for d in depth:
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=d)
        clf = clf.fit(x_train, y_train)
        y_test_hat = clf.predict(x_test)  # 测试数据
        result = (y_test_hat == y_test)  # True则预测正确，False则预测错误
        #np.mean(result)  这个是准确率的均值 ，用1 去减掉这个 就是错误率
        #这里 错误率 看文档
        err = 1 - np.mean(result)
        err_list.append(err)
        print d, ' 准确度: %.2f%%' % (100 * err)
    plt.figure(facecolor='w')
    #画一条 深度  和错误率 的线
    plt.plot(depth, err_list, 'ro-', lw=2)
    plt.xlabel(u'决策树深度', fontsize=15)
    plt.ylabel(u'错误率', fontsize=15)
    plt.title(u'决策树深度与过拟合', fontsize=17)
    plt.grid(True)
    plt.show()
