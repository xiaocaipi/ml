# -*- coding:utf-8 -*-
#get the mnist data
# wget http://deeplearning.net/data/mnist/mnist.pkl.gz



# 数据 是写了 黑白的 0到9 的数字  是mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True)

import tensorflow as tf

# Parameters   参数的设置
learning_rate = 0.001
training_epochs = 30
#每次计算的量
batch_size = 100
display_step = 1

# Network Parameters  设置第1层神经元的个数
n_hidden_1 = 256 # 1st layer number of features
# 设置第2层神经元的个数
n_hidden_2 = 512 # 2nd layer number of features
# 这里mnist 的一个小图 是 28×28  是一个784 的向量
n_input = 784 # MNIST data input (img shape: 28*28)
#  10个类
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input   类型是float的    [None, n_input]  设置 shape  这里None  可以设置成100
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Create model
#x 是输入的值  weights  和biases  都是参数
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    #因为是传统的神经网络 就用传统的矩阵运算   x和weight 先矩阵相乘  然后和biases 相加
    #这个weights 是事先定义好的矩阵
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    #然后接入一个relu
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    #下一层的神经网络 还是矩阵相乘 相加  用的是上一层的输出
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    #进行非线性激励
    layer_2 = tf.nn.relu(layer_2)

    #we can add dropout layer
    # drop_out = tf.nn.dropout(layer_2, 0.75)


    # Output layer with linear activation
    #最后是输出   weights['out']  是生成n个class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & biases
#h1  h2  设置 矩阵的shape  用的是Variable  是可以直接求导的
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
#biases 都是一些一个维度的向量
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
#构建 感知机
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
#label  是y   中间值是pred   计算出cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#用AdamOptimizer  给的 learning_rate  对cost 求最小
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
#tf init
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))