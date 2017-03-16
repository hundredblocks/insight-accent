import tensorflow as tf
import numpy as np


class OneDimensionalNetworks():
    def __init__(self, n_channels, n_samples):
        self.n_channels = n_channels
        self.n_samples = n_samples

    def transform(self, content, style):
        content_tf = np.ascontiguousarray(content.T[None, None, :, :])
        style_tf = np.ascontiguousarray(style.T[None, None, :, :])
        return content_tf, style_tf

    def to_spectrogram(self, x):
        return x[0, 0].T

    def generate_input(self, placeholder=True):
        if placeholder:
            return tf.placeholder('float32', [1, 1, self.n_samples, self.n_channels], name="x")
        return tf.Variable(np.random.randn(1, 1, self.n_samples, self.n_channels).astype(np.float32) * 1e-3, name="x")


class SingleLayerConv(OneDimensionalNetworks):
    def __init__(self, filter_width, n_channels, n_samples, n_filters):
        OneDimensionalNetworks.__init__(self, n_channels, n_samples)
        std = np.sqrt(2) * np.sqrt(2.0 / ((n_channels + n_filters) * filter_width))
        # filter shape is "[filter_height, filter_width, in_channels, out_channels]"
        self.kernel_values = np.random.randn(1, filter_width, n_channels, n_filters) * std

    def get_feature(self, x):
        kernel_tf = tf.constant(self.kernel_values, name="kernel", dtype='float32')

        conv = tf.nn.conv2d(
            x,
            kernel_tf,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")

        content = tf.nn.relu(conv)
        style = content
        return content, style


class DoubleLayerConv(OneDimensionalNetworks):
    def __init__(self, filter_width, n_channels, n_samples, n_filters):
        OneDimensionalNetworks.__init__(self, n_channels, n_samples)
        std = np.sqrt(2) * np.sqrt(2.0 / ((n_channels + n_filters) * filter_width))
        self.kernel1_values = np.random.randn(1, filter_width, n_channels, n_filters) * std
        self.kernel2_values = np.random.randn(1, filter_width, n_filters, int(n_filters // 2)) * std

    def get_feature(self, x):
        kernel_tf = tf.constant(self.kernel1_values, name="kernel1", dtype='float32')
        kernel_tf2 = tf.constant(self.kernel2_values, name="kernel2", dtype='float32')

        conv = tf.nn.conv2d(
            x,
            kernel_tf,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")

        style = tf.nn.relu(conv)
        conv2 = tf.nn.conv2d(
            style,
            kernel_tf2,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv2")

        content = tf.nn.relu(conv2)

        return content, style


class TwoDimensionalNetworks():
    def __init__(self, n_samples, filter_width, n_channels, n_filters):
        self.filter_width = filter_width
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.n_samples = n_samples

    def transform(self, content, style):
        content_tf = np.ascontiguousarray(content[None, :, :, None])
        style_tf = np.ascontiguousarray(style[None, :, :, None])
        return content_tf, style_tf

    def to_spectrogram(self, x):
        return x[0][:, :, 0]

    def generate_input(self, placeholder=True):
        if placeholder:
            return tf.placeholder('float32', [1, self.n_channels, self.n_samples, 1], name="x")
        return tf.Variable(np.random.randn(1, self.n_channels, self.n_samples, 1).astype(np.float32) * 1e-3, name="x")


class SingleLayer2DConv(TwoDimensionalNetworks):
    def __init__(self, filter_height=7, filter_width=7, n_channels=1, n_samples=130, n_filters=32):
        TwoDimensionalNetworks.__init__(self, n_samples, filter_width, n_channels, n_samples)
        std = np.sqrt(2) * np.sqrt(2.0 / ((n_channels + n_filters) * filter_width))
        self.kernel1_values = np.random.randn(filter_height, filter_width, 1, n_filters) * std

    def get_feature(self, x):
        kernel_tf = tf.constant(self.kernel1_values, name="kernel1", dtype='float32')

        conv = tf.nn.conv2d(
            x,
            kernel_tf,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")

        content = tf.nn.relu(conv)
        style = content

        return content, style
