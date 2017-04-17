import logging

import librosa
import numpy as np
import tensorflow as tf
from IPython.display import Audio, display

from models.classifier import SoundCNN, conv2d
from utils import read_audio_spectrum, fft_to_audio, plot_all

N_FFT = 2048
N_FILTERS = 2048
FILTER_WIDTH = 11
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def preprocess_samples(content, style, n_fft=N_FFT, reduce_factor=1):
    content_filename = "inputs/" + content
    style_filename = "inputs/" + style

    x_c, fs_c = librosa.load(content_filename)
    x_s, fs_s = librosa.load(style_filename)
    if fs_c != fs_s:
        raise ValueError("Different sampling frequencies for style %s and content %s" % (fs_s, fs_c))
    a_content = read_audio_spectrum(x_c, fs_c, n_fft=n_fft, reduce_factor=reduce_factor)
    a_style = read_audio_spectrum(x_s, fs_s, n_fft=n_fft, reduce_factor=reduce_factor)

    # reducing to same size
    n_samples = min(a_content.shape[1], a_style.shape[1])
    n_channels = min(a_content.shape[0], a_style.shape[0])
    a_style = a_style[:n_channels, :n_samples]
    a_content = a_content[:n_channels, :n_samples]

    a_content_tf = np.ascontiguousarray(a_content.T[None, None, :, :])
    a_style_tf = np.ascontiguousarray(a_style.T[None, None, :, :])

    return a_content_tf, a_style_tf, n_samples, n_channels, fs_s, a_content, a_style


def style_transfer(num_classes, n_samples, n_channels, content_tensor, style_tensor, a_content, a_style):
    alpha = 1e-2
    with tf.Session() as session:
        model = SoundCNN(num_classes)
        saver = tf.train.Saver()
        saver.restore(session, 'trained_model/model.ckpt')
        x_np = np.random.randn(1, 1, n_samples, n_channels).astype(np.float32) * 1e-3
        x = tf.Variable(x_np, name="x")
        content_features = model.h_conv1.eval(feed_dict={model.x: content_tensor,
                                                         model.keep_prob: 1.0,
                                                         model.is_train: False})
        style_features = model.h_conv1.eval(feed_dict={model.x: style_tensor,
                                                       model.keep_prob: 1.0,
                                                       model.is_train: False})
        n_filters = style_features.shape[-1]
        features = np.reshape(style_features, (-1, n_filters))
        style_gram = np.matmul(features.T, features) / n_samples
        logging.log(logging.INFO, "Style gram")
        logging.log(logging.INFO, style_gram)

        conv1 = conv2d(x, model.W_conv1)
        batch_norm1 = tf.contrib.layers.batch_norm(conv1,
                                                   center=True, scale=True,
                                                   is_training=False,
                                                   )
        h_conv1 = tf.nn.relu(batch_norm1)

        end = h_conv1
        style = h_conv1

        content_loss = alpha * 2 * tf.nn.l2_loss(end - content_features)

        _, height, width, number = map(lambda i: i.value, style.get_shape())
        print(style.get_shape())
        feats = tf.reshape(style, (-1, number))
        gram = tf.matmul(tf.transpose(feats), feats) / n_samples
        style_loss = 2 * tf.nn.l2_loss(gram - style_gram)

        loss = content_loss + style_loss

        opt = tf.contrib.opt.ScipyOptimizerInterface(
            loss, method='L-BFGS-B', options={'maxiter': 300}, var_list=[x])

        tf.global_variables_initializer().run()

        initial_gram = gram.eval()
        initial_content = end.eval()
        initial_vector = x.eval()
        start_loss = loss.eval()

        logging.info('Start loss: %s', start_loss)
        logging.info('Started optimization.')
        opt.minimize(session)
        logging.info('Final loss: %s', loss.eval())

        end_gram = gram.eval()
        end_content = end.eval()

        result = x.eval()

        logging.info('style_difference')
        logging.info(end_gram - initial_gram)

        logging.info('content_difference')
        logging.info(end_content - initial_content)

        final_result = np.zeros_like(a_content)
        final_result[:n_channels, :] = np.exp(result[0, 0].T) - 1

        initial_spectrogram = np.zeros_like(a_content)
        initial_spectrogram[:n_channels, :] = np.exp(initial_vector[0, 0].T) - 1
        plot_all(a_content, a_style, final_result, initial_spectrogram)

        return final_result


def do_style_transfer(content, style, num_classes=2, n_fft=N_FFT, reduce_factor=1):
    content_tensor, style_tensor, n_samples, n_channels, sampling_frequency, a_content, a_style = preprocess_samples(
        content,
        style,
        n_fft=n_fft,
        reduce_factor=reduce_factor)
    result = style_transfer(num_classes, n_samples, n_channels, content_tensor, style_tensor, a_content, a_style)
    out_name = 'styled.wav'
    output_filename = fft_to_audio(out_name, result, sampling_frequency)
    display(Audio(output_filename))


if __name__ == '__main__':
    do_style_transfer(content="op_cancelled.wav", style="british_test.wav", num_classes=2, n_fft=N_FFT, reduce_factor=1)
