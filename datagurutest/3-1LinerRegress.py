# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
#画图工具
import matplotlib.pyplot as plt

# In[6]:

# 使用numpy生成200个随机点   从-0.5 到0.5 范围 产生200个点
#因为用到的是二维的数据，所以加一个维度，是一个200 行一列的数据
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
#生成一些干扰项，形状和x_data 是一样的
noise = np.random.normal(0, 0.02, x_data.shape)
#y_data 设置为 x 平方 加干扰项     x 和y  形成一个 开口向上的抛物线
y_data = np.square(x_data) + noise

# 定义两个placeholder   [None, 1] 定义形状  列是只有1列 行不确定
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

#输入一个x 经过神经网络的计算 最后会得到一个y ，这个y 是预测值，  y‘  和真实的y 比较接近
#因为x 是一个点 输入层 是只有一个神经元，中间层是可以调整的，用10 个神经元 作为中间层，输出层也是1个点
# 定义神经网络中间层，定义权值，刚开始给一个随机数，形状是1行10列
#权值 是连接 输入层和 中间层  输入层是一个神经元  ，中间层是10个神经元  所以形状是 1  10
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
#偏置值  也初始化，初始化为0 ，因为有10 个神经元 ，所以 有10个偏置值
biases_L1 = tf.Variable(tf.zeros([1, 10]))
#x 输入是个矩阵，权值 也是个矩阵  乘起来，再加上偏置值  这样就能得到信号的总和
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
#用 tanh 做为激活函数 得到l1   l1 是中间层的输出
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络输出层
#输出层 和上面差不多 神经元个数需要改变  l2 这一层的权值 的形状是 10 1   输出层只有1个神经元
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
#因为输出层只有一个神经元 所以 偏置值 只有一个
biases_L2 = tf.Variable(tf.zeros([1, 1]))
#信号的总和  是l1 和Weights_L2  相乘 加上 l2 的偏置值
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
#最后的预测还需要经过一个激活函数
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 代价函数 还是用二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        reslut=sess.run(train_step, feed_dict={x: x_data, y: y_data})
        if step % 20 == 0:
            print (step)

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 画图0
    plt.figure()
	#首先把样本 点画出来
    plt.scatter(x_data, y_data)
	#把预测的值打出来，用红色 -  ，线的宽度 为5
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()


# In[ ]:



