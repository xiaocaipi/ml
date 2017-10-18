# coding: utf-8

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense] = 1
  return labels_one_hot

path = u'/home/menu/project/stock/test.txt'
n_batch=100

np.set_printoptions(suppress=True)
data = np.loadtxt(path, dtype=float, delimiter='\t')
# batch_size = data.shape
y, x = np.split(data, (6,),axis=1)

y = y[:, 5:6]


n = y.shape[0]
labels_dense =np.reshape(y,(n,))
labels_dense=labels_dense.astype(np.int64)
y = dense_to_one_hot(labels_dense,10)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
batch_size= np.shape(y_train)[0] /n_batch



def next_batch(current_batch,n_batch):
    start =0
    stop =0
    if current_batch==0:
        start =0
        stop =start+n_batch
    else:
        start =  current_batch*n_batch
        stop = start +n_batch
    return x_train[start:stop],y_train[start:stop]

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
    x1 = tf.placeholder(tf.float32, [None, 100],name='x-input')
    y1 = tf.placeholder(tf.float32, [None, 10],name='y-input')

with tf.name_scope('layer'):
    with tf.name_scope('wights'):
        # W = tf.Variable(tf.zeros([100, 10]),name='W')
        W = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1),name='W')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10])+0.1, name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x1, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.relu(wx_plus_b)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y1,logits=prediction))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

init = tf.global_variables_initializer()

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y1, 1), tf.argmax(prediction, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('../logs/', sess.graph)
    for epoch in range(20):

        # batch_xs, batch_ys = next_batch(epoch, n_batch)
        # summary, _ = sess.run([merged, train_step], feed_dict={x1: batch_xs, y1: batch_ys})
        # writer.add_summary(summary, epoch)
        # acc = sess.run(accuracy, feed_dict={x1: x_test, y1: y_test})
        # print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
        b
        for batch in range(batch_size):
            batch_xs, batch_ys =next_batch(batch,n_batch)
            # batch_xs= np.reshape(batch_xs,(100,100)).astype(np.float32)
            # batch_ys = np.reshape(batch_ys, (100, 10)).astype(np.float32)
            # sess.run(train_step, feed_dict={x1: batch_xs, y1: batch_ys})
            summary, a,b = sess.run([merged, train_step,prediction], feed_dict={x1: batch_xs, y1: batch_ys})
            b= b[1:2]

        writer.add_summary(summary, epoch)
        acc = sess.run(accuracy, feed_dict={x1: x_test, y1: y_test})
        # print b
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc)+",prediction")