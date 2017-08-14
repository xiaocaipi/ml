import tensorflow as tf

W1 = tf.Variable(tf.truncated_normal([10,2],stddev=0.1))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    result=sess.run(W1)

    print result
