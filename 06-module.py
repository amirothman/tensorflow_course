# Create a Simple TB Graph

logdir = '/tmp/demo'
a = tf.constant(3,name='a')
b = tf.constant(4,name='b')
c = tf.multiply(a,b,name='multiply')
d = tf.div(a, b, name='divide')
sess = tf.Session()
writer = tf.summary.FileWriter(logdir)
writer.add_graph(sess.graph)
print(sess.run(c))

# Before running TensorBoard, you need to generate
# summary data in a log directory by creating a summary writer.

writer = tf.summary.FileWriter(logdir)
writer.add_graph(sess.graph)

# or

writer = tf.summary.FileWriter(logdir, sess.graph)

# Node Names
# Node Scope
import tensorflow as tf

with tf.name_scope('multiply'):
    c = tf.multiply(a, b, name='c')

with tf.name_scope('divide'):
    d = tf.div(a, b, name='d')


# Merge and Output Summary

# Merge all the summary operations
merged_summary = tf.summary.merge_all()
s = sess.run(merged_summary)

# Output all the summary operations to TB
writer.add_summary(s)

# MNIST NN on Tensorboard

tf.summary.scalar("Loss", loss)
tf.summary.scalar("Accuracy", accuracy)

tf.summary.histogram('W1', W1)
tf.summary.histogram('B1', B1)
tf.summary.histogram('W2', W2)
tf.summary.histogram('B3', B3)
tf.summary.histogram('W5', W5)
tf.summary.histogram('B5', B5)
