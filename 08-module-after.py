import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp", one_hot=True)

# Parameters

learning_rate = 0.001
epochs = 50000
batch_size = 128

# Network parameters

n_input = 28
n_steps = 28

n_hidden = 128
n_classes = 10 # 0.. 9

# Placeholders for input data

x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = tf.Variable(tf.random_normal([n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

output_layer = {
    "weights": tf.Variable(tf.random_normal([n_hidden, n_classes])),
    "biases": tf.Variable(tf.random_normal([n_classes]))
    }

def recurrent_neural_network_model(x, weights, biases):
    
    # current data input shape: (batch_size, n_steps, n_input)
    # required shape: n_steps tensors list of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # LSTM cell
    lstm_cell = rnn.BasicLSTMCell(n_hidden)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.add(tf.matmul(outputs[-1], output_layer["weights"]), output_layer["biases"])

prediction = recurrent_neural_network_model(x, weights, biases)

# Define cost-function and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Evaluate
correct_predictions = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step * batch_size < epochs:
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # reshape data to get 28 sequence of 28 elements

        batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        if step % 20 == 0:
            acc = sess.run(accuracy, feed_dict = {x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict = {x: batch_x, y: batch_y})
            print("accuracy:", acc)
            print("loss:", loss)

        step += 1

    test_len = 128 # testing 128 images
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    testing_accuracy = sess.run(accuracy, feed_dict={x: test_data, y:test_label})

    print("test_accuracy:", testing_accuracy)
