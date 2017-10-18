# coding: utf-8

# In[2]:

import tensorflow as tf
# 手写数字相关的工具包
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# In[3]:

# 载入数据集  第一个参数是个路劲，这里放在当前路径，one_hot  就是把标签转为 0 和1的形式
mnist = input_data.read_data_sets("../data/mnist", one_hot=True)

# 每个批次的大小，训练模型的时候 不是1张1张 放到神经网络里面去训练的，是一次性放入一个批次
# 会以矩阵的形式放进去
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
# 只有2个层 一个是输入层 一个是输出层  #输入层 是784 个神经元   #输出层是10个标签，10个神经元
# 权值 是连接输入层和输出层
W = tf.Variable(tf.zeros([784, 10]))
# 偏置值 是连接输出层
b = tf.Variable(tf.zeros([10]))
# tf.matmul(x,W)+b  是信号的总和
# 信号的总和在经过一个softmax 函数，输出的信号 都转换为一个概率值，这里的softmax 就是一个激活函数
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数
#loss = tf.reduce_mean(tf.square(y - prediction))
#用对数似然函数 作为代价函数，真实值 是y  预测值是prediction
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 测试模型的准确率
# 结果存放在一个布尔型列表中
# argmax返回一维张量中最大的值所在的位置
# 那 真实的y 返回的位置   和 预测返回的位置 ，看是不是一个类别
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率
# 把 correct_prediction  格式转换下，correct_prediction 是布尔类型 转成 float 32，然后再求一个 平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    # 所有的图片循环21次
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))


# In[ ]:



