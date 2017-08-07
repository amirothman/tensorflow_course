import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

# Constants
a = tf.constant(1.8)
b = tf.constant(2.0, tf.float32)
c = tf.constant(3, dtype=tf.float32)

# Session

with tf.Session() as sess:
	print(sess.run(a))
	print(sess.run(b))
	print(sess.run(c))

# Data type casting

a = tf.constant(1.8,dtype=tf.float32)
b = tf.cast(a, tf.int32)


# Math Operations

b = tf.cast(b, tf.float32)

tf.add(a,b)
tf.subtract(a,b)
tf.multiply(a,b)
tf.div(a,b)
tf.truediv(a, b)
tf.floordiv(a, b)
tf.mod(a, b)
tf.square(2)
tf.sqrt(4.0)
tf.sin(3.1416)
tf.tan(3.1416)
tf.cos(3.1416)
tf.exp(1.0)

# Basic Matrix Operations

a = tf.constant([[1,2],[3,4],[5,6]])
b = tf.constant([[1,5],[3,2],[5,1]])
d = tf.random_normal([5,5])
tf.add(a,b)
c = tf.transpose(a)
tf.matmul(a,c)
tf.matrix_inverse(d)

a = tf.constant([[1,1,1],[2,2,2]])

# Matrix sum operation
tf.reduce_sum(a)	# sum all the elements
tf.reduce_sum(a, 0)	# column sums
tf.reduce_sum(a, 1)	# row sums

# Special Matrices

tf.zeros([2,3])
tf.ones([2,3])
tf.diag(np.ones(2))
tf.fill([2,3],2)
tf.eye(3)

# Matrix Operations

x = [[1,1]]
w = [[1,2],[3,4]]
b = [[2,2]]

# y = x*w+b

# Random Numbers

tf.random_normal([1])
tf.truncated_normal([2,3])

tf.random_normal(shape=[2, 2])

seed = 42
tf.random_uniform([1])
tf.set_random_seed(seed)

# Placeholders serve as inputs or parameters

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

sum_ = tf.add(a,b)
with tf.Session() as sess:
	print(sess.run(sum_,feed_dict={a:3,b:4}))

# For the previous exercise, replace x as in input
#
# x = [[1,1]]
# w = [[1,2],[3,4]]
# b = [[2,2]]
#
# y = x*w+b

# Variables are used for training the adjustable parameters in a graph

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.1], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# Initialize global variable
with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	print(sess.run(linear_model, {x:[1,2,3,4]}))
