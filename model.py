# architecture from https://github.com/awjuliani/sound-cnn

import tensorflow as tf
import numpy as np


class SoundCNN():
    def __init__(self, num_classes):
        self.x = tf.placeholder(tf.float32, [None, 1, 130, 1025])
        self.y_ = tf.placeholder(tf.float32, [None, num_classes])

        self.W_conv1 = weight_variable([1, 7, 1025, 32])

        # self.b_conv1 = bias_variable([32])
        # self.h_conv1 = tf.nn.relu(conv2d(self.x, self.W_conv1) + self.b_conv1)

        conv1 = conv2d(self.x, self.W_conv1)
        self.batch_norm1 = tf.contrib.layers.batch_norm(conv1,
                                                        center=True, scale=True,
                                                        is_training=True,
                                                        )
        self.h_conv1 = tf.nn.relu(self.batch_norm1)
        self.h_pool1 = max_pool_2x2(self.h_conv1)

        self.W_conv2 = weight_variable([5, 5, 32, 64])

        # self.b_conv2 = bias_variable([64])
        # self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2)

        conv2 = conv2d(self.h_pool1, self.W_conv2)
        self.batch_norm2 = tf.contrib.layers.batch_norm(conv2,
                                                        center=True, scale=True,
                                                        is_training=True)
        self.h_conv2 = tf.nn.relu(self.batch_norm2)

        self.h_pool2 = max_pool_2x2(self.h_conv2)

        self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 3 * 11 * 64])
        self.W_fc1 = weight_variable([3 * 11 * 64, 1024])

        # self.b_fc1 = bias_variable([1024])
        # self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        conv3 = tf.matmul(self.h_pool2_flat, self.W_fc1)
        self.batch_norm3 = tf.contrib.layers.batch_norm(conv3,
                                                        center=True, scale=True,
                                                        is_training=True)

        self.h_fc1 = tf.nn.relu(self.batch_norm3)

        self.keep_prob = tf.placeholder("float")
        self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
        self.W_fc2 = weight_variable([1024, num_classes])
        self.b_fc2 = bias_variable([num_classes])

        self.h_fc2 = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
        self.y_conv = tf.nn.softmax(tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2)

        self.cross_entropy = -tf.reduce_sum(self.y_ * tf.log(tf.clip_by_value(self.y_conv, 1e-10, 1.0)))
        self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
