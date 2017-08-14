
# coding: utf-8

# In[2]:
#这里例子 是怎么把这个准确率 提升到98%
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
keep_prob=tf.placeholder(tf.float32)
#多定义一个学习率的变量  初始的是0.001
lr = tf.Variable(0.001, dtype=tf.float32)

#创建一个简单的神经网络
#定义2个隐藏层 第一个是500 个神经元
W1 = tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
b1 = tf.Variable(tf.zeros([500])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1,keep_prob)
#第2个隐藏层是300 个神经元
W2 = tf.Variable(tf.truncated_normal([500,300],stddev=0.1))
b2 = tf.Variable(tf.zeros([300])+0.1)
#激活函数是 tanh
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.truncated_normal([300,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L2_drop,W3)+b3)

#交叉熵代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#训练 用的是adadelta  把lr 学习率 放进去 作为参数
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
	#迭代51 个周期
    for epoch in range(51):
		#先储率 学习率 ，没迭代一个周期 ，学习率 会等于 0.001 *0.95 的迭代次数的次方    ** 是次方的意思
		#让学习率 随着 迭代的次数 逐渐的减小，这样的好处是，一开始模型是比较混乱的，希望可以给一个比较大的学习率，可以快速的收敛
		#当找到一个局部最小值 或者全局最小值，当需要朝着这个全局最小值靠近的时候，把学习率主键降低，这样才能 到达最小值
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
			#训练的时候  keep_prob:1.0  就相当于没有用 dropout
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        #把 lr 给run 一下， run一下 就相当于得到 这个lr的值
        learning_rate = sess.run(lr)
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print ("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc) + ", Learning Rate= " + str(learning_rate))


# In[ ]:



