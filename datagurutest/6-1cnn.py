# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# In[2]:

mnist = input_data.read_data_sets('../data/mnist', one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size


# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图


# 定义一个函数 来初始化权值
# 把权值的形状传过来，进行一个初始化
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 生成一个截断的正态分布
    return tf.Variable(initial, name=name)


# 初始化偏置
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# 定义一个卷积层，一个卷积的操作，使用的是tf 库里面 的conv2d
# conv2d  的意思 是一个二维的操作
def conv2d(x, W):
    # 看一下参数
    # 首先是 x   x是一个tensor 形状是 [batch, in_height, in_width, in_channels] 这样的形状，
    # tensor 是一个4维的，第一个维度 batch 是一个批次大小，有一个长和宽 ，比如对第一层 就是输入层，输入层就是图片 ，这个长和宽就是图片的，in_channels 是通道数，一个黑白的图片的通道数就是1 ，彩色的就是3
    # 第二个参数是 W w 是一个滤波器，或者说是卷积核  filter / kernel   形状也是一个 tensor 形状是 [filter_height, filter_width, in_channels, out_channels]  是 滤波器的长 和宽  滤波器输入的通道数 和输出的通道数
    # 第三个参数是步长`strides[0] = strides[3] = 1` 第0 和3 个位置都是1 . strides[1]代表x方向的步长，strides[2]代表y方向的步长
    # padding 有2种: A `string` from: `"SAME", "VALID"`
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义池化层，这里用max pooling
def max_pool_2x2(x):
    # ksize [1,x,y,1]
    # 参数和上面的 conv2d 差不多 ，x 是一样的，ksize 是窗口的大小  第0  3 个位置也是要设置1  中间2个代表的是 2*2 的窗口
    # strides 是步长 第0 3 都是1  ，x 方向的步长是 2   y方向的步长也是2
    # padding 方式和场面一样
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 命名空间
with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope('x_image'):
        # x 传进来是 ？*784  因为要对图片进行 卷积和池化的操作，图片是2维的，所以要转成2维的
        # 这个x_image 就是给上面 卷积和池化的x
        # 改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]`   -1 代表批次的大小 ，这里等一下会设置成100 ，-1 是代表？
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

# 初始化 第一个卷积层的权值 和偏置值
with tf.name_scope('Conv1'):
    # 初始化第一个卷积层的权值和偏置
    with tf.name_scope('W_conv1'):
        # 权值的形状穿进去的形状是[5,5,1,32]，5*5 代表是卷积采样的窗口，1 是代表输入的通道是多少  最后32 代表的输出是32个特征平面，32 个卷积核从平面抽取特征，最后会有32个特征图
        W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')
    with tf.name_scope('b_conv1'):
        # 每一个卷积核一个偏置值  这里用到了32个卷积核 ，就用到32个偏置值
        b_conv1 = bias_variable([32], name='b_conv1')

    # 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        # 把卷积得到的操作 传入pooling
        h_pool1 = max_pool_2x2(h_conv1)  # 进行max-pooling
# 定义第二个卷积层
with tf.name_scope('Conv2'):
    # 初始化第二个卷积层的权值和偏置
    with tf.name_scope('W_conv2'):
        # 还是用5*5 的采样窗口，前面的1 这里是32，因为 经过卷积层1 之后 生成了32个特征平面图，原来就只有1个黑白的图，  64 是输出是64个特征平面
        W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64], name='b_conv2')  # 每一个卷积核一个偏置值

    # 把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)  # 进行max-pooling

# 原来的一张图片是 28*28的图片，经过第一次卷积后还是28*28，第一次池化后变为14*14
# 第二次卷积后为14*14，第二次池化后变为了7*7
# 进过上面操作后得到64张7*7的特征平面
# h_pool2 tensor 的形状是 100 7  7 64    100 是批次

# 加上2个全连接层
with tf.name_scope('fc1'):
    # 初始化第一个全连接层的权值
    with tf.name_scope('W_fc1'):
        # 因为是全连接层 因此只有2个维度  7*7*64  相当是 前面h_pool2 的结果 是有64张7*7 的平面，就全连接层的输入 是连接到 7*7*64 个神经元，然后1024  是全连接层本身有1024 个神经元
        W_fc1 = weight_variable([7 * 7 * 64, 1024], name='W_fc1')  # 上一层有7*7*64个神经元，全连接层有1024个神经元
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024], name='b_fc1')  # 1024个节点

    # 把池化层2的输出扁平化为1维
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name='h_pool2_flat')
    # 求第一个全连接层的输出
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(wx_plus_b1)

    # keep_prob用来表示神经元的输出概率  设置dropout
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

# 定义第二个全连接层
with tf.name_scope('fc2'):
    # 初始化第二个全连接层
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024, 10], name='W_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10], name='b_fc2')
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    with tf.name_scope('softmax'):
        # 计算输出
        prediction = tf.nn.softmax(wx_plus_b2)

# 交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
                                   name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

# 使用AdamOptimizer进行优化
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 求准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔列表中
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test', sess.graph)
    for i in range(1001):
        # 训练模型
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
        # 记录训练集计算的参数
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        train_writer.add_summary(summary, i)
        # 记录测试集计算的参数
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, i)

        if i % 100 == 0:
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images[:10000], y: mnist.train.labels[:10000],
                                                      keep_prob: 1.0})
            print ("Iter " + str(i) + ", Testing Accuracy= " + str(test_acc) + ", Training Accuracy= " + str(train_acc))



# In[ ]:



