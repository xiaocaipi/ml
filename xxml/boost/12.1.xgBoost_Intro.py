# /usr/bin/python
# -*- encoding:utf-8 -*-

import xgboost as xgb
import numpy as np

# 1、xgBoost的基本使用
# 2、自定义损失函数的梯度和二阶导
# 3、binary:logistic/logitraw


# 定义f: theta * x
def log_reg(y_hat, y):
    p = 1.0 / (1.0 + np.exp(-y_hat))
    g = p - y.get_label()
    h = p * (1.0-p)
    return g, h


def error_rate(y_hat, y):
    return 'error', float(sum(y.get_label() != (y_hat > 0.5))) / len(y_hat)


if __name__ == "__main__":
    # 读取数据
    #这个data_train 是xgb 也就是 xgboost 自定义的对象
    data_train = xgb.DMatrix('12.agaricus_train.txt')
    data_test = xgb.DMatrix('12.agaricus_test.txt')

    # 设置参数
    #每一个树的最大深度是2  是非常的浅了，这个eta 就是衰减因子   silent  是不是在树生成过程中输出给我们
    #objective  指出是要做分类 回归，这里是要做一个二分类问题   如果是多分类就是softmax
    param = {'max_depth': 2, 'eta': 1, 'silent': 0, 'objective': 'binary:logitraw'} # logitraw
    #这里用 逻辑回归,xgboost 需要给定一阶导和二阶导，给了逻辑回归，二分类，softmax 等 就是给了一阶导和二阶导的信息
    # param = {'max_depth': 3, 'eta': 0.3, 'silent': 1, 'objective': 'reg:logistic'}
    #定义 测试数据  训练数据
    watchlist = [(data_test, 'eval'), (data_train, 'train')]
    #做几次的计算，也就是树的个数
    n_round = 3
    #bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)
    #给训练数据，额外要给一些参数
    bst = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist, obj=log_reg, feval=error_rate)

    # 计算错误率
    #做预测
    y_hat = bst.predict(data_test)
    #获取测试数据的标记  即使实际的y
    y = data_test.get_label()
    print y_hat
    print y
    error = sum(y != (y_hat > 0))
    error_rate = float(error) / len(y_hat)
    print '样本总数：\t', len(y_hat)
    print '错误数目：\t%4d' % error
    # 这里就用3 棵树  深度是2  就有非常好的效果
    # 是过拟合了，为什么深度学习在图像上有好的效果，因为参数达到了很多，因为就是过拟合，过拟合能过在测试数据和训练数据上有一个好的结果。
    print '错误率：\t%.5f%%' % (100*error_rate)
