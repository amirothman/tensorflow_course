# Load Iris Flower Dataset
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


from sklearn import datasets

iris = datasets.load_iris()
data = iris.data
target = iris.target

print(data)
print(target)

# One hot encoding
import numpy as np

a = [0,1,2,1]
num_labels = len(np.unique(a))
b = np.eye(num_labels)[a]
print(b)

import tensorflow as tf
# To decode the one-hot matrix, can use tf.argmax.
# tf.argmax - return the position of the max value

c = tf.argmax(b,axis=1)

# Split Dataset to Train/Test Sets

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(data, target, test_size=0.33, random_state=42)
