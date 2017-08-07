import tensorflow as tf

# Step 1: Initial Setup

# Setup the placeholders and variables

X = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_normal([]),tf.float32)
b = tf.Variable([0.1],tf.float32)

# Step 2: Define a Model

yhat = tf.multiply(W,X) + b
y = tf.placeholder(tf.float32)

# Step 3: Define Loss Function

# A loss (cost) function measures how the error of the prediction from the known answer
#
# Eg Mean Square Error

loss = tf.reduce_sum(tf.square(yhat - y))

# Step 4: Define Optimizer

# The Optimizer used gradient-based algorithm to reduce the loss (error) during training
learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(loss)

# Step 5: Training Loop
from tflearn.datasets import cifar10
(train_X, train_y), (X_test, y_test) = cifar10.load_data()
# Iterate the optimizer to reduce error
with tf.Session() as sess:
    for i in range(1000):
        sess.run(tf.global_variables_initializer())
        sess.run(train, {X:train_X, y:train_y})


# Step 1: Initial Setup (MNIST)

X = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Step 2: Define a Model (MNIST)

yhat = tf.nn.softmax(tf.matmul(X,W)+b)
y = tf.placeholder(tf.float32, [None, 10])

# Step 3: Define Loss Function

# Cross Entropy

loss = -tf.reduce_sum(y*tf.log(yhat))

# Step 4: Define Optimizer

optimizer = tf.train.GradientDescentOptimizer(learning_rate)

train = optimizer.minimize(loss)

# Step 5: Training Loop
# Train mini batch of 100 pages for each iteration (epoch)

for i in range(1000):
    batch_X, batch_y = mnist.train.next_batch(100)
    train_data = {X: batch_X, y: batch_y}
    sess.run(train, feed_dict=train_data)

# Step 6: Evaluation

is_correct = tf.equal(tf.argmax(y,1),tf.argmax(yhat,1))

accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

sess.run(train, feed_dict=test_data)
sess.run(accuracy,feed_dict = test_data)

# Softmax Cross Entropy LF

tf.nn.softmax_cross_entropy_with_logits(labels=y, logits = yhat)

# Replace the previous model with yhat = tf.matmul(X, W) + b
# and loss function with loss = tf.reduce_mean(   tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
#
# Try out different learning rates:
# 0.001. 0.01. 0.1, 0.5

yhat = tf.matmul(X, W) + b
loss = tf.reduce_mean(   tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
