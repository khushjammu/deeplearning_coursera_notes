from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# hyperparameters
learning_rate = 0.5



mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data')#, one_hot=True)

n_x = mnist.train.images.shape[1]
print("N_X IS EQUAL TO {}".format(n_x))

# implement forward propagation
x = tf.placeholder(tf.float32, shape = [None, 784], name = 'x')
W = tf.Variable(tf.zeros([784, 10]), name = 'W')
b = tf.Variable(tf.zeros([10]), name = 'b')
y_hat = tf.matmul(x, W) + b

y = tf.placeholder(tf.int64, shape = [None])

# implement cost function
#cost = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_hat)
cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_hat)

# implement gradient descent
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# run gradient descent
for i in range(1000):
	batch_x_train, batch_y_train = mnist.train.next_batch(100)
	sess.run([train_step], feed_dict = {x:batch_x_train, y:batch_y_train})
	if (i % 100)==0:
		print("On iteration {}.".format(i))


# test trained model
correct_prediction = tf.equal(tf.argmax(y_hat,1), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(
	accuracy, feed_dict={
	x:mnist.test.images,
	y:mnist.test.labels
	}))










