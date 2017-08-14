# coding: utf-8


# 看tensorboard  网络运行 ，在5-2 的程序上拿过来修改
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# In[2]:


mnist = input_data.read_data_sets("../data/mnist", one_hot=True)

batch_size = 100

n_batch = mnist.train.num_examples // batch_size


# 定义一个函数 ，计算参数的值，参数概要
# 传入一个参数 var  计算这个var的平均值，标准差，最大值，最小值，和直方图
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        # 记录mean 这个值 在给一个名字
        tf.summary.scalar('mean', mean)  # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('layer'):
    # 创建一个简单的神经网络
    with tf.name_scope('wights'):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        # 这个权值 需要在网络的运行过程中 去分析权值的变化 ，就用和这个variable_summaries 函数去分析下
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    # 看loss的变化
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 看准确率的变化
        tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary，把前面 scalar的指标 要合并一下
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('../logs/', sess.graph)
    for epoch in range(51):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 在训练的时候 把merged  要加上，就每次训练一次 就去统计一下 merged，然后跑完是返回 [merged,train_step] ，把这个 用summary,_  来接收
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})

        # 把summary 记录到 文件里面去，写的内容一个是 summary  一个是 epoch
        # 这里每一个epoch 去写到文件里面去
        writer.add_summary(summary, epoch)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))


# In[ ]:

# for i in range(2001):
#     #m每个批次100个样本
#     batch_xs,batch_ys = mnist.train.next_batch(100)
#     summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
#     writer.add_summary(summary,i)
#     if i%500 == 0:
#         print(sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels}))

