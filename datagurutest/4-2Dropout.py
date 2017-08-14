
# coding: utf-8

# In[3]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("../data/mnist",one_hot=True)

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
#在定义一个placeholder，用这个ph 来设置一些 dropout的参数
keep_prob=tf.placeholder(tf.float32)

#创建一个简单的神经网络
#参数一开始用0 做初始化 并不好
#给权值初始化的时候 用 截断的正太分布来初始化， 然后stddev  标准差是0.1
#tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
#截断正态分布随机数，均值mean,标准差stddev,不过只保留[mean-2*stddev,mean+2*stddev]范围内的随机数
#这里中间层 设置2000 个神经元，是一个复杂的网络,实际上不需要那么复杂的网络，这里变复杂的话 想让出现过拟合的情况，一般出现过拟合 是因为 网络太复杂，然后数据太少
#出现了过拟合 然后用dropout 来试一下
#
W1 = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
#给偏置值 初始化 0 +0.1, 用这样的w 和b 的初始化 方式效果会好一点，初始化 也是一个比较重要的
b1 = tf.Variable(tf.zeros([2000])+0.1)
#激活函数用tanh
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
#定义一个 L1的 dropout 正则化项 ，调用tf  的dropout 函数，把L1 穿进去，keep_prob 是设置百分之多少的神经元是工作的
#keep_prob  是1 的话 说明 百分之百的神经元工作  ，0.5 就50%
L1_drop = tf.nn.dropout(L1,keep_prob)

#l2 隐藏层  还是2000个神经元
W2 = tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2,keep_prob)

#l3 隐藏层   1000个神经元
W3 = tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)
L3_drop = tf.nn.dropout(L3,keep_prob)

#输出 10个 神经元
W4 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L3_drop,W4)+b4)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)


init = tf.global_variables_initializer()


correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(31):
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
			#在训练的时候 用到 dropout 是0.7
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})
        #测试数据的准确率 ，测试的时候 用的dropout 1.0 让所有的神经元的都工作
        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
		#训练集的准确率  这个keep_prob  是1.0 用全部神经元的话  会过拟合
        train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) +",Training Accuracy " + str(train_acc))


# In[ ]:
#结果  用的是2000个神经元  训练数据的准确率 很高到99    测试数据的准确率到 97%    keep_prob 是1.0 的情况
#Iter 29,Testing Accuracy 0.9727,Training Accuracy 0.995655
#Iter 30,Testing Accuracy 0.9722,Training Accuracy 0.995782
#结果  用的是2000个神经元  训练数据的准确率 很高到99    测试数据的准确率到 97%    keep_prob 是0.7 的情况
#收敛速度会变慢，但是可以防止过拟合，测试数据准确率 和 训练数据准确率 差距比较大的话，可以看做是过拟合
#Iter 30,Testing Accuracy 0.971,Training Accuracy 0.977
