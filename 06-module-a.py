import tensorflow as tf

logdir = '/tmp/demo'
a = tf.constant(3,name='a')
b = tf.constant(4,name='b')
c = tf.multiply(a,b,name='multiply')
d = tf.div(a, b, name='divide')
sess = tf.Session()
writer = tf.summary.FileWriter(logdir)
writer.add_graph(sess.graph)
print(sess.run(c))
