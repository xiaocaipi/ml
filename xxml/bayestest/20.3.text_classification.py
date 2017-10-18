#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from time import time
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib as mpl

#clf 是某一个分类器
def test_clf(clf):
    print u'分类器：', clf
    #某些分类器 是需要 alpha 的，比如多项式朴素贝叶斯 就需要一个alpha 的值
    #这里 alpha 是10 的-3 次方 到2次方   一共取了10 个
    alpha_can = np.logspace(-3, 2, 10)
    #5折 的交叉验证  如果只有一个 alpha 超参数的话  10 个 alpha ，做了10次 测试 ，这里做了5折，所以是做了50次
    model = GridSearchCV(clf, param_grid={'alpha': alpha_can}, cv=5)
    m = alpha_can.size
    #如果分类器 有 alpha 属性 就给一个 
    if hasattr(clf, 'alpha'):
        model.set_params(param_grid={'alpha': alpha_can})
        m = alpha_can.size
    #如果是k 均值的话 是有一个超参数 n   n_neighbors
    if hasattr(clf, 'n_neighbors'):
        #这个跨度从1 到14
        neighbors_can = np.arange(1, 15)
        model.set_params(param_grid={'n_neighbors': neighbors_can})
        # 算了多少次
        m = neighbors_can.size
    #svm 需要c 作为调参
    if hasattr(clf, 'C'):
        # 这里从 10 到 100 取了3 个数
        C_can = np.logspace(1, 3, 3)
        #gamma  也是3个
        gamma_can = np.logspace(-3, 0, 3)
        model.set_params(param_grid={'C':C_can, 'gamma':gamma_can})
        m = C_can.size * gamma_can.size
    #随机森林
    if hasattr(clf, 'max_depth'):
        max_depth_can = np.arange(4, 10)
        model.set_params(param_grid={'max_depth': max_depth_can})
        m = max_depth_can.size
    t_start = time()
    model.fit(x_train, y_train)
    t_end = time()
    #1次训练的 时间 m是 一共的次数 5是 5折交叉验证
    t_train = (t_end - t_start) / (5*m)
    print u'5折交叉验证的训练时间为：%.3f秒/(5*%d)=%.3f秒' % ((t_end - t_start), m, t_train)
    print u'最优超参数为：', model.best_params_
    t_start = time()
    y_hat = model.predict(x_test)
    t_end = time()
    t_test = t_end - t_start
    print u'测试时间：%.3f秒' % t_test
    acc = metrics.accuracy_score(y_test, y_hat)
    print u'测试集准确率：%.2f%%' % (100 * acc)
    name = str(clf).split('(')[0]
    index = name.find('Classifier')
    if index != -1:
        name = name[:index]     # 去掉末尾的Classifier
    if name == 'SVC':
        name = 'SVM'
    return t_train, t_test, 1-acc, name


if __name__ == "__main__":
    print u'开始下载/加载数据...'
    t_start = time()
    # remove = ('headers', 'footers', 'quotes')
    remove = ()
    #关注 这个类别
    categories = 'alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space'
    # categories = None     # 若分类所有类别，请注意内存是否够用
    #去下载 数据  指定是要训练数据 还是测试数据  需要哪里写类被   remove  是对一些标题 一些脚码 删除工作，这里是什么都不删除
    data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=0, remove=remove)
    data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=0, remove=remove)
    t_end = time()
    print u'下载/加载数据完成，耗时%.3f秒' % (t_end - t_start)
    #这个是一个dataset 的数据结构
    print u'数据类型：', type(data_train)
    #有2034 个 训练数据 和 1353 个测试数据
    print u'训练集包含的文本数目：', len(data_train.data)
    print u'测试集包含的文本数目：', len(data_test.data)
    #这里是有4个类别
    print u'训练集和测试集使用的%d个类别的名称：' % len(categories)
    #把4个类别的名称输出一下
    categories = data_train.target_names
    #用pprint 打印出来效果会好一些
    pprint(categories)
    # 训练数据的 类别
    y_train = data_train.target
    #测试数据的类别
    y_test = data_test.target
    #先把数据 前面10个看一下
    print u' -- 前10个文本 -- '
    for i in np.arange(10):
        print u'文本%d(属于类别 - %s)：' % (i+1, categories[y_train[i]])
        print data_train.data[i]
        print '\n\n'
    #用 tfidf  去提取特征  用英语的停止词  比如 the    max_df 是文档频率的词 的最大频率，有的时候这个会很大 这里限制0.5 最大
    vectorizer = TfidfVectorizer(input='content', stop_words='english', max_df=0.5, sublinear_tf=True)
    #获取向量化之后的数据
    x_train = vectorizer.fit_transform(data_train.data)  # x_train是稀疏的，scipy.sparse.csr.csr_matrix
    x_test = vectorizer.transform(data_test.data)
    #训练集样本个数：2034，特征个数：33809   特征个数 就是建立的词典 的个数
    print u'训练集样本个数：%d，特征个数：%d' % x_train.shape
    print u'停止词:\n',
    pprint(vectorizer.get_stop_words())
    feature_names = np.asarray(vectorizer.get_feature_names())

    print u'\n\n===================\n分类器的比较：\n'
    clfs = (MultinomialNB(),                # 0.87(0.017), 0.002, 90.39%
            BernoulliNB(),                  # 1.592(0.032), 0.010, 88.54%
            KNeighborsClassifier(),         # 19.737(0.282), 0.208, 86.03%
            RidgeClassifier(),              # 25.6(0.512), 0.003, 89.73%
            RandomForestClassifier(n_estimators=200),   # 59.319(1.977), 0.248, 77.01%
            SVC()                           # 236.59(5.258), 1.574, 90.10%
            )
    result = []
    for clf in clfs:
        #把分类器带到test_clf 里面去  得到返回结果 a  
        a = test_clf(clf)
        # 放到result  里面去
        result.append(a)
        print '\n'
    #最后去做可视化
    result = np.array(result)
    time_train, time_test, err, names = result.T
    x = np.arange(len(time_train))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 7), facecolor='w')
    ax = plt.axes()
    b1 = ax.bar(x, err, width=0.25, color='#77E0A0')
    ax_t = ax.twinx()
    b2 = ax_t.bar(x+0.25, time_train, width=0.25, color='#FFA0A0')
    b3 = ax_t.bar(x+0.5, time_test, width=0.25, color='#FF8080')
    plt.xticks(x+0.5, names, fontsize=10)
    leg = plt.legend([b1[0], b2[0], b3[0]], (u'错误率', u'训练时间', u'测试时间'), loc='upper left', shadow=True)
    # for lt in leg.get_texts():
    #     lt.set_fontsize(14)
    plt.title(u'新闻组文本数据不同分类器间的比较', fontsize=18)
    plt.xlabel(u'分类器名称')
    plt.grid(True)
    plt.tight_layout(2)
    plt.show()
