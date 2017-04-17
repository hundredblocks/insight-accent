import itertools
import logging

import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython.display import Audio, display

from models import random_models
from utils import read_audio_spectrum, fft_to_audio, plot_all

N_FFT = 2048
N_FILTERS = 4096
FILTER_WIDTH = 11
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train_and_output(content, style, n_fft=N_FFT, n_filters=N_FILTERS, filter_width=FILTER_WIDTH, reduce_factor=1):
    content_filename = "inputs/" + content
    content_no_extention = content.split(".")[0]
    style_no_extention = style.split(".")[0]
    style_filename = "inputs/" + style

    x_c, fs_c = librosa.load(content_filename)
    x_s, fs_s = librosa.load(style_filename)
    a_content = read_audio_spectrum(x_c, fs_c, n_fft=n_fft, reduce_factor=reduce_factor)
    a_style = read_audio_spectrum(x_s, fs_s, n_fft=n_fft, reduce_factor=reduce_factor)

    n_samples = min(a_content.shape[1], a_style.shape[1])
    logging.info("content samples %s" % a_content.shape[1])
    logging.info("style samples %s" % a_style.shape[1])
    n_channels = min(a_content.shape[0], a_style.shape[0])

    # Truncate style to content frequency and time window (debatable)
    a_style = a_style[:n_channels, :n_samples]
    a_content = a_content[:n_channels, :n_samples]

    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
        # data shape is "[batch, in_height, in_width, in_channels]",
        # model = random_models.DoubleLayerConv(filter_width, n_channels, n_samples, n_filters)
        model = random_models.SingleLayerConv(filter_width, n_channels, n_samples, n_filters)
        # model = random_models.SingleLayer2DConv(7, 7, n_channels, n_samples, 32)
        x = model.generate_input(placeholder=True)

        content_layer, feature_layer = model.get_feature(x)
        a_content_tf, a_style_tf = model.transform(a_content, a_style)

        content_features = content_layer.eval(feed_dict={x: a_content_tf})
        style_features = feature_layer.eval(feed_dict={x: a_style_tf})
        n_filters = style_features.shape[-1]
        features = np.reshape(style_features, (-1, n_filters))
        style_gram = np.matmul(features.T, features) / n_samples

    result, initial = train(n_samples, model, content_features, style_gram)

    initial_spectrogram = np.zeros_like(a_content)
    initial_spectrogram[:n_channels, :] = np.exp(model.to_spectrogram(initial)) - 1
    final_spectrogram = np.zeros_like(a_content)
    final_spectrogram[:n_channels, :] = np.exp(model.to_spectrogram(result)) - 1

    # This code is supposed to do phase reconstruction
    out_name = '%s_to_%s_%s_fft_%s_width_%s_n.wav' % (
        content_no_extention, style_no_extention, n_fft, filter_width, n_filters)
    output_filename = fft_to_audio(out_name, final_spectrogram, fs_c)
    display(Audio(output_filename))
    plot_all(a_content, a_style, final_spectrogram, initial_spectrogram)


def train(n_samples, model, content_features, style_gram):
    alpha = 1e-2

    with tf.Graph().as_default():
        # Initial soundwave
        x = model.generate_input(placeholder=False)

        content_layer, feature_layer = model.get_feature(x)
        content_loss = alpha * 2 * tf.nn.l2_loss(content_layer - content_features)

        _, height, width, number = map(lambda i: i.value, feature_layer.get_shape())
        print(feature_layer.get_shape())
        feats = tf.reshape(feature_layer, (-1, number))
        gram = tf.matmul(tf.transpose(feats), feats) / n_samples
        style_loss = 2 * tf.nn.l2_loss(gram - style_gram)

        # Overall loss
        loss = content_loss + style_loss

        opt = tf.contrib.opt.ScipyOptimizerInterface(
            loss, method='L-BFGS-B', options={'maxiter': 300})

        # Optimization
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            initial = x.eval()

            print('Started optimization.')
            print 'Start loss:', loss.eval()
            opt.minimize(sess)

            print 'Final loss:', loss.eval()
            result = x.eval()
        return result, initial


def inspect_audio(output_filename, content_spectral, style_spectral, result_spectral):
    display(Audio(output_filename))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Content')
    plt.imshow(content_spectral[:400, :])
    plt.subplot(1, 3, 2)
    plt.title('Style')
    plt.imshow(style_spectral[:400, :])
    plt.subplot(1, 3, 3)
    plt.title('Result')
    plt.imshow(result_spectral[:400, :])
    plt.show()


def hyperparameter_grid_search(content, style, fft_values, filter_size_values, n_filter_values):
    params = itertools.product(fft_values, filter_size_values, n_filter_values)
    for config in params:
        train_and_output(content, style, n_fft=config[0], filter_width=config[1], n_filters=config[2])


if __name__ == '__main__':
    train_and_output(content="manu_american.wav", style="manu_british.wav", n_fft=2048, filter_width=3, n_filters=4096)
