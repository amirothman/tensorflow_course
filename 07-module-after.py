import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# Load MNIST data

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp", one_hot=True)

# Parameters

learning_rate = 0.001
epoch = 200000
batch_size = 128

# Network Parameters

n_input = 784 # image shape: 28*28
n_classes = 10 # 0..9
dropout = 0.75

# Inputs of computation graph

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
    # 5x5 convolution, 1 inputs, 32 outputs
    "wc1": tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 convolution, 32 inputs, 64 outputs
    "wc2": tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    "wd1": tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outsputs (prediction of classes)
    "out": tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
            "bc1": tf.Variable(tf.random_normal([32])),
            "bc2": tf.Variable(tf.random_normal([64])),
            "bd1": tf.Variable(tf.random_normal([1024])),
            "out": tf.Variable(tf.random_normal([n_classes]))
         }

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

# Defining the models

def conv_net(x, weights, biases, dropout):
    
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution layer
    conv1 = conv2d(x, weights["wc1"], biases["bc1"])
    conv1 = maxpool2d(conv1, k = 2)

    # another Convolution layer
    conv2 = conv2d(conv1, weights["wc2"], biases["bc2"])
    conv2 = maxpool2d(conv2, k=2)

    # fully connected layer

    fully_connected = tf.reshape(conv2, [-1, weights["wd1"].get_shape().as_list()[0]])
    fully_connected = tf.add(tf.matmul(fully_connected, weights["wd1"]), biases["bd1"])
    fully_connected = tf.nn.relu(fully_connected)
    fully_connected = tf.nn.dropout(fully_connected, dropout)

    out = tf.add(tf.matmul(fully_connected, weights["out"]), biases["out"])
    
    return out


dropout_tensor = tf.placeholder(tf.float32)
predictions = conv_net(x, weights, biases, dropout_tensor)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_predictions = tf.equal(tf.argmax(predictions,1), tf.argmax(y,1))

accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step * batch_size < epoch:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y, dropout_tensor: dropout})

        if step % 20 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict = {x: batch_x, y: batch_y, dropout_tensor: 1})

            print("loss:", loss)
            print("accuracy:", acc)

        step += 1

    print("Testing accuracy")

    print(sess.run(accuracy, feed_dict = {x: mnist.test.images[:256], y:mnist.test.labels[:256], dropout_tensor: 1}))
        
