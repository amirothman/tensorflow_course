
import tensorflow as tf

# Common Activation Functions

x = tf.random_normal([5,5])
tf.nn.relu(x)
tf.nn.sigmoid(x)
tf.nn.softmax(x)
tf.nn.tanh(x)

# Common Optimizer in TF
learning_rate = 0.001
tf.train.GradientDescentOptimizer(learning_rate)
tf.train.AdamOptimizer(learning_rate)

# Step 1: Initial Setup (MNIST)

X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# NN Architecture

L1 = 200
L2 = 100
L3 = 60
L4 = 30

# Initialize the weights and biases with truncated normal distribution
W1 = tf.Variable(tf.truncated_normal([784, L1], stddev=0.1))
B1 = tf.Variable(tf.truncated_normal([L1], stddev=0.1))

# Truncated normal follows a normal distribution with specified
# mean and standard deviation, except that values whose magnitude
# is more than 2 standard deviations from the mean are dropped and re-picked.

B2 = tf.Variable(tf.zeros([L2]))
W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
B3 = tf.Variable(tf.zeros([L3]))
W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
B4 = tf.Variable(tf.zeros([L4]))
W5 = tf.Variable(tf.truncated_normal([L4, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

# Step 2: NN Model

Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
yhat = tf.nn.softmax(Ylogits)

# Step 3: Loss Function

loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=y)

# Step 4: Optimizer

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(loss)

# Step 5: Training Loop

for epoch in range(10):
    for i in range(550):
        batch_X, batch_y = mnist.train.next_batch(100)
        train_data = {X: batch_X, y: batch_y}
        sess.run(train, feed_dict=train_data)
        print("Training Accuracy = ", sess.run(accuracy, feed_dict=train_data))

# Step 6: Evaluation

test_data = {X:mnist.test.images,y:mnist.test.labels}
sess.run(accuracy, feed_dict = test_data)

# Saving Model

saver = tf.train.Saver()
saver.save(sess, model_file)

# Restoring Model

saver.restore(sess, model_file)
