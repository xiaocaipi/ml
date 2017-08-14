# -*- coding:utf-8 -*-
import tensorflow as tf
#定义一个常量  op   一行2列
m1=tf.constant([[3,3]])
#定义一个常量  op   2行1列
m2=tf.constant([[2],[3]])
#创建一个矩阵乘法的op  matmul是矩阵的乘法  把m1  和m2传入
product = tf.matmul(m1,m2)
#Tensor("MatMul:0", shape=(1, 1), dtype=int32)
#结果是一个tensor ，形状是1 ×1 的就是一个数字   类型是32 位的整形
#   因为现在只定义了一些op  没有定义图 和session，需要图在session 中运行
print product
#定义一个会话，启动默认图  定义的会话有一个默认的图，所以图就不用再去定义
sess= tf.Session()
#再会话中 使用run方法 去执行product，再执行product 的时候 会去调用矩阵的乘法，tf.matmul(m1,m2)
#调用矩阵乘法的时候 会再去生成m1 m2 2个常量，一层一层往上调用
result= sess.run(product)
#[[15]]
print  result
#关闭会话
sess.close()
#一般定义一个会话比较麻烦，会这么定义,这样就不需要执行关闭操作
with tf.Session() as sess:
    result = sess.run(product)
    print  result


#变量讲解
#定义一个 变量
x=tf.Variable([1,2])
#定义一个常量
a=tf.constant([3,3])
#定义一减法的op
sub=tf.subtract(x,a)
#定义一加法的op
add=tf.add(x,sub)

#变量需要 初始化一下
#全局变量的初始化，初始化所有变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    #再会话中 先run 下init
    sess.run(init)
    result1 = sess.run(sub)
    result2 = sess.run(add)
    #[-2 -1]
    print  result1
    #[-1  1]
    print  result2


#写一个循环，让变量加1
#定义一个 变量，初始化时候 是0 起一个名字 叫counter
state=tf.Variable(0,name='counter')
#定义一个加法的op  让state变量加1
new_value=tf.add(state,1)
#调用赋值的操作，在tf里面，在会话中，赋值 不能用等号 来赋值，需要调用assign 赋值的方法
#把 new_value  给state
update=tf.assign(state,new_value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #刚开始 state 是0
    print(sess.run(state))
    for _ in range(5):
        #update 是赋值操作
        print (sess.run(update))




#介绍tf 里面的fetch  和feed

#fetch 概念  就是再会话里面 同时执行多个op  得到运行的结果
#先定义3个常量
input1=tf.constant(3.0)
input2=tf.constant(2.0)
input3=tf.constant(5.0)

add= tf.add(input2,input3)
mu1=tf.multiply(input1,add)

with tf.Session() as sess:
    #这里会话运行多个op  一个乘法，一个加法
    result4=sess.run([mu1,add])
    #[21.0, 7.0]   2个op 的结果
    print result4

#说下一下 feed 概念
#定义2个占位符  传入类型  这里是一个32 位的浮点
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
#乘法操作 ，input1  和input2  但是不像前面定义的常数，这里并没有确认 input1 的值
#这个乘法的值可以在运行乘法op的时候 传入
output=tf.multiply(input1,input2)

#占位符可以再会话中调用使用
with tf.Session() as sess:
    #现在要运行output  再把值传入，传入值用到的是 字典的形式   input1 传入的是7.0  input2 传入的是2.0
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))





#tf  来一个简答的示例
#还需要 numpy
import numpy as np

#生成 100 个随机的点
x_data=np.random.rand(100)
#这个公式就相当与是一个直线
y_data=x_data*0.1+0.2

#构造一个现行模型
#一个初始值 0. 表明是 浮点型
b=tf.Variable(0.)
k=tf.Variable(0.)
#这个线性模型的 截距是一个变量  斜率也是一个变量 2个的 初始值都是0
y=k*x_data+b

#定义一个二次代价函数
#reduce_mean  是求平均值的意思
#y_data  是真实值，y是预测值，   square  是求平方，误差的平方 然后在求一个平均值
loss=tf.reduce_mean(tf.square(y_data-y))
#定义一个梯度下降法来进行训练的优化器
#在tf 里面把优化的方法 叫做优化器,这里用最简单梯度下降优化器,给一个0.2的学习率
optimizer= tf.train.GradientDescentOptimizer(0.2)
#定义一个最小化 代价函数,训练的目的 就是为了最小化 这个代价函数
train=optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #进行迭代  这里迭代201次
    #每次迭代 就去run  定义的train op  run train的时候  回去 最小化 loss   这个loss 就是 误差平方平均值
    #误差的话 y_data 已经是确定的值，而y 是k*x_data+b   需要 k  和b 这个是2个 变量
    #再tf 里面可以用 GradientDescentOptimizer  去优化  k和b
    for step in range(201):
        sess.run(train)
        #每20次打印一下  步数 和k 和b
        if step %20==0:
            print (step,sess.run([k,b]))
