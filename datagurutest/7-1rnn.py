# coding: utf-8

# In[1]: rnn  一般用在文本 语音，用在图片处理也是可以的

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# In[2]:

# 载入数据集
mnist = input_data.read_data_sets("../data/mnist", one_hot=True)
# 有一个输入层  一个隐藏层 一个输出层  隐藏层的输出 会输入到下一个时间，进行综合的计算
# 输入图片是28*28
# 输入层是28个神经元
n_inputs = 28  # 输入一行，一行有28个数据
# 把图片看做一个序列，这个图片有28次输入，每一次是输入 28个数据  一共输入28次
max_time = 28  # 一共28行
# 隐藏层单元的个数 ，相当于隐藏层有多少个神经元  在lstm 网络中不叫神经元叫 block，这里有100个block
lstm_size = 100  # 隐层单元
n_classes = 10  # 10个分类
batch_size = 50  # 每批次50个样本
n_batch = mnist.train.num_examples // batch_size  # 计算一共有多少个批次

# 这里的none表示第一个维度可以是任意的长度
x = tf.placeholder(tf.float32, [None, 784])
# 正确的标签
y = tf.placeholder(tf.float32, [None, 10])

# 初始化权值 权值矩阵 是100 *10
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
# 初始化偏置值
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


# 定义RNN网络
# 第一个参数是x  ，x 就是 训练的数据
# 第二个参数是权值  第三个是偏置值
def RNN(X, weights, biases):
    # inputs=[batch_size, max_time, n_inputs]
    # 数据的转换  因为一个批次是50  所以 x 是50  * 784 的  所以x 需要转换成 -1  ，这个-1 就是50    max_time 是28   n_inputs 是28
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    # 定义LSTM基本CELL   就是block   定义100 个 中间隐藏层的个数
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    # final_state[0]是cell state  是block  中间的信号
    # final_state[1]是hidden_state  是block  最后的信号
    # 进行计算  会得到2个返回值  一个是 output  一个是final_state ，这个是rnn 最后的输出
    # final_state  有3个维度  final_state[state,batch_size,cell.state_size]
    # 第一个维度 state 这里有2个值 一个是0 一个是1  0 是cell state   1 是hidden state
    # 第二个维度  batch size  就是一个批次 多少个样本 就是有50
    # 第三个 维度 是state size  隐藏单元的个数  就是100
    # 还有 一个返回是output    这个和 time_major  有关，如果time_major  是false 默认是false，数据的格式是  [batch_size, max_time, cell.output_size]
    # 如果是true  [max_time, batch_size, cell.output_size]
    # hidden state 和output的区别就是  hidden state 记录的是 这个时间序列 最后一次的输出，这里时间序列是28个序列，一幅图 会向lstm 网络中传28次，hs 记录的是最后一次的输出
    #	而output记录的是每一次的输出结果，有一个 max_time，如果max_time=1  就是事件序列 第一次的结果， 如果 max_time=2 的时候 就是 第二次的结果   是从0 到27
    # max_time  最后一次的输出结果  就是和hidden state 的输出结果一样
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    # 这里用输出 final_state[1]  去乘以权值  在加上一个偏置值 ，这个就是神经网络最后的输出
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results


# 计算RNN的返回结果
prediction = RNN(x, weights, biases)
# 损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 把correct_prediction变为float32类型
# 初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print ("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))


# In[ ]:



