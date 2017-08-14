# coding: utf-8

# In[1]:
# 讲tensorboard 可视化功能
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 多导入了一个 projector 的包
from tensorflow.contrib.tensorboard.plugins import projector

# In[2]:

# 载入数据集
mnist = input_data.read_data_sets("../data/mnist", one_hot=True)
# 运行次数
max_steps = 1001
# 图片数量
image_num = 3000
# 文件路径,里面有一个图片，图片有1万个数字，每一个数字就是一个手写数字
# 在这个下面 要新建 一个projector 文件夹，在 projector 下面 要有一个data  和一个 projector  文件夹，然后把手写数字的图片放在data 下面  projector 下面是空的
DIR = "/home/menu/PycharmProjects/deeplearning/data/"

# 定义会话
sess = tf.Session()

# 载入图片，把图片保存在 embedding 里面，用的是一个变量
# 用的一个stack 方法 有一个 'x' is [1, 4]  'y' is [2, 5] 'z' is [3, 6]
# stack([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.   就相当于把 xyz 打包起来
# stack([x, y, z], axis=1) => [[1, 2, 3], [4, 5, 6]]
# 这里就相当于把测试集里面的0 到image_num 张图片 打一个包，然后存放在 embedding
embedding = tf.Variable(tf.stack(mnist.test.images[:image_num]), trainable=False, name='embedding')


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


# 命名空间
with tf.name_scope('input'):
    # 这里的none表示第一个维度可以是任意的长度
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    # 正确的标签
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

# 显示图片  tensorboard 里面有一栏 image ，把训练后图片显示出来
with tf.name_scope('input_reshape'):
    # 把x的形状转为[-1, 28, 28, 1] 这样的形状  -1 代表不确定的值   因为一开始的x 的行也是不确定的  是一个28*28 的 把x的784 写成28*28 的 后面的1 说明是黑白的  3是彩色的
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    # 用 image 方法 把image_shaped_input 传进去，一共放10张图片
    tf.summary.image('input', image_shaped_input, 10)

with tf.name_scope('layer'):
    # 创建一个简单神经网络
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784, 10]), name='W')
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, W) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

with tf.name_scope('loss'):
    # 交叉熵代价函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    # 使用梯度下降法
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 初始化变量
sess.run(tf.global_variables_initializer())

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔型列表中
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 把correct_prediction变为float32类型
        tf.summary.scalar('accuracy', accuracy)

# 产生metadata文件 ，就在定义的 dir 下面有没有projector/projector/metadata.tsv 这个文件
if tf.gfile.Exists(DIR + 'projector/projector/metadata.tsv'):
    # 有的话就删除掉
    tf.gfile.DeleteRecursively(DIR + 'projector/projector/metadata.tsv')
# 去生成这样的文件
with open(DIR + 'projector/projector/metadata.tsv', 'w') as f:
    # 先去得到测试集里面的标签
    labels = sess.run(tf.argmax(mnist.test.labels[:], 1))
    # 循环3000次
    for i in range(image_num):
        # 把图片的label 写入 metadata 里面去
        f.write(str(labels[i]) + '\n')

        # 合并所有的summary
merged = tf.summary.merge_all()

# 定义一个writer
projector_writer = tf.summary.FileWriter(DIR + 'projector/projector', sess.graph)
# 定义一个saver 来保存网络的图形的
saver = tf.train.Saver()
# 定义了一个配置文件  是个固定操作
config = projector.ProjectorConfig()
# 固定操作
embed = config.embeddings.add()
# 把embedding 的名字 给 embed 的tensor name
embed.tensor_name = embedding.name
# 把embed.metadata_path 赋值
embed.metadata_path = DIR + 'projector/projector/metadata.tsv'
# 还要给图片的路径，就是之前放在data 下的图片
embed.sprite.image_path = DIR + 'projector/data/mnist_10k_sprite.png'
# 图片切分 按照28*28 像素的切分
embed.sprite.single_image_dim.extend([28, 28])
# 把projector_writer 放进去  配置也放进去
projector.visualize_embeddings(projector_writer, config)

# 迭代max_steps次
for i in range(max_steps):
    # 每个批次100个样本
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 固定的写法
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # 固定的写法
    run_metadata = tf.RunMetadata()
    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys}, options=run_options,
                          run_metadata=run_metadata)
    # 记录参数的变化
    projector_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    projector_writer.add_summary(summary, i)
    # 每训练100 次打印准确率
    if i % 100 == 0:
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print ("Iter " + str(i) + ", Testing Accuracy= " + str(acc))
# 全部训练好，把训练好的模型保存下来
saver.save(sess, DIR + 'projector/projector/a_model.ckpt', global_step=max_steps)
# 关闭操作
projector_writer.close()
sess.close()


# In[ ]:



