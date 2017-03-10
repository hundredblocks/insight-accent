import librosa
import numpy as np
import tensorflow as tf
from IPython.display import Audio, display
from utils import read_audio_spectrum

N_FFT = 2048


def train_output(content, style, reduce_factor=1):
    content_filename = "inputs/" + content
    content_no_extention = content.split(".")[0]
    style_no_extention = style.split(".")[0]
    style_filename = "inputs/" + style

    x_c, fs_c = librosa.load(content_filename)
    x_s, fs_s = librosa.load(style_filename)
    a_content, fs = read_audio_spectrum(x_c, fs_c, reduce_factor=reduce_factor)
    a_style, fs = read_audio_spectrum(x_s, fs_s, reduce_factor=reduce_factor)

    n_samples = a_content.shape[1]
    n_channels = a_content.shape[0]

    # Truncate style to content frequency and time window (debatable)
    a_style = a_style[:n_channels, :n_samples]

    n_filters = 4096

    a_content_tf = np.ascontiguousarray(a_content.T[None, None, :, :])
    a_style_tf = np.ascontiguousarray(a_style.T[None, None, :, :])

    # filter shape is "[filter_height, filter_width, in_channels, out_channels]"
    filter_width = 11
    std = np.sqrt(2) * np.sqrt(2.0 / ((n_channels + n_filters) * filter_width))
    kernel = np.random.randn(1, filter_width, n_channels, n_filters) * std

    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        # data shape is "[batch, in_height, in_width, in_channels]",
        x = tf.placeholder('float32', [1, 1, n_samples, n_channels], name="x")

        kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
        conv = tf.nn.conv2d(
            x,
            kernel_tf,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")

        net = tf.nn.relu(conv)

        content_features = net.eval(feed_dict={x: a_content_tf})
        style_features = net.eval(feed_dict={x: a_style_tf})

        features = np.reshape(style_features, (-1, n_filters))
        style_gram = np.matmul(features.T, features) / n_samples

    result = train(n_samples, n_channels, kernel, content_features, style_gram)

    a = np.zeros_like(a_content)
    a[:n_channels, :] = np.exp(result[0, 0].T) - 1

    # This code is supposed to do phase reconstruction
    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
    for i in range(500):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, N_FFT))

    output_filename = 'outputs/' + content_no_extention + '_' + style_no_extention + '.wav'
    librosa.output.write_wav(output_filename, x, fs)

    print output_filename
    display(Audio(output_filename))

    # plt.figure(figsize=(15, 5))
    # plt.subplot(1, 3, 1)
    # plt.title('Content')
    # plt.imshow(a_content[:400, :])
    # plt.subplot(1, 3, 2)
    # plt.title('Style')
    # plt.imshow(a_style[:400, :])
    # plt.subplot(1, 3, 3)
    # plt.title('Result')
    # plt.imshow(a[:400, :])
    # plt.show()


def train(n_samples, n_channels, kernel, content_features, style_gram):
    ALPHA = 1e-2
    learning_rate = 1e-3
    iterations = 100

    result = None
    with tf.Graph().as_default():
        # Build graph with variable input
        #     x = tf.Variable(np.zeros([1,1,N_SAMPLES,N_CHANNELS], dtype=np.float32), name="x")
        x = tf.Variable(np.random.randn(1, 1, n_samples, n_channels).astype(np.float32) * 1e-3, name="x")

        kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
        conv = tf.nn.conv2d(
            x,
            kernel_tf,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")

        net = tf.nn.relu(conv)

        content_loss = ALPHA * 2 * tf.nn.l2_loss(net - content_features)

        style_loss = 0

        _, height, width, number = map(lambda i: i.value, net.get_shape())

        size = height * width * number
        feats = tf.reshape(net, (-1, number))
        gram = tf.matmul(tf.transpose(feats), feats) / n_samples
        style_loss = 2 * tf.nn.l2_loss(gram - style_gram)

        # Overall loss
        loss = content_loss + style_loss

        opt = tf.contrib.opt.ScipyOptimizerInterface(
            loss, method='L-BFGS-B', options={'maxiter': 300})

        # Optimization
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            print('Started optimization.')
            opt.minimize(sess)

            print 'Final loss:', loss.eval()
            result = x.eval()
        return result


if __name__ == '__main__':
    train_output(style="op_cancelled.wav", content="wake_up.wav", reduce_factor=1)
