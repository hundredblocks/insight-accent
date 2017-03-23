"""from https://github.com/pkmital/tensorflow_tutorials/blob/master/python/09_convolutional_autoencoder.py
Tutorial on how to create a convolutional autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""

import matplotlib
import numpy as np
import tensorflow as tf

from autoencoder_models import VAE, VAE_MNIST

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_fetch import get_male_female_pairs
from utils import fft_to_audio


def test(mnist_flag=True):
    """Summary

    Returns
    -------
    name : TYPE
        Description
    """
    # %%
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt

    # %%
    # Fit all training data
    t_i = 0
    batch_size = 100
    n_epochs = 10
    n_examples = 20

    # %%
    # load MNIST as before
    if mnist_flag:
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        test_xs, _ = mnist.test.next_batch(n_examples)
        xs, ys = mnist.test.images, mnist.test.labels
        num_examples = mnist.train.num_examples
        hidden_size = 2
        ae = VAE_MNIST(n_hidden=hidden_size)

    else:
        data_and_path, fs = get_male_female_pairs('encoder_data/DAPS/small_test/cut', product=False)
        only_female = [f[1] for f in data_and_path]
        train, val, test_xs = split_dataset(only_female, .1, .1)
        tdim = only_female[0].shape[0]
        fdim = only_female[0].shape[1]
        num_examples = len(train)
        ae = VAE(input_shape=[None, tdim * fdim])

    # hidden_size = 2
    # ae = VAE(n_hidden=hidden_size)

    # %%
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    fig_manifold, ax_manifold = plt.subplots(1, 1)
    fig_reconstruction, axs_reconstruction = plt.subplots(2, n_examples, figsize=(10, 2))
    fig_image_manifold, ax_image_manifold = plt.subplots(1, 1)
    for epoch_i in range(n_epochs):
        print('--- Epoch', epoch_i)
        train_cost = 0
        print("Number train examples %s" % num_examples)
        print("Num batches %s" % (num_examples // batch_size))
        if not mnist_flag:
            perms = np.random.permutation(train)
        for batch_i in range(num_examples // batch_size):

            if mnist_flag:
                batch_xs, _ = mnist.train.next_batch(batch_size)
            else:
                batch_xs = perms[batch_i * batch_size:(batch_i + 1) * batch_size, :]

            train_cost += sess.run([ae['cost'], optimizer],
                                   feed_dict={ae['x']: batch_xs})[0]
        # %%
        # Plot example reconstructions from walking the latent layer
        imgs = []
        for img_i in np.linspace(-3, 3, n_examples):
            for img_j in np.linspace(-3, 3, n_examples):
                z = np.array([[img_i, img_j]], dtype=np.float32)
                recon = sess.run(ae['y'], feed_dict={ae['z']: z})
                imgs.append(np.reshape(recon, (1, 28, 28, 1)))
        imgs_cat = np.concatenate(imgs)
        ax_manifold.imshow(montage_batch(imgs_cat))
        fig_manifold.savefig('images/manifold_%08d.png' % t_i)

        # %%
        # Plot example reconstructions
        recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs})
        print("reconstruction obtained")
        for example_i in range(n_examples):
            axs_reconstruction[0][example_i].imshow(
                np.reshape(test_xs[example_i, :], (28, 28)),
                cmap='gray')
            axs_reconstruction[1][example_i].imshow(
                np.reshape(
                    np.reshape(recon[example_i, ...], (784,)),
                    (28, 28)),
                cmap='gray')
            axs_reconstruction[0][example_i].axis('off')
            axs_reconstruction[1][example_i].axis('off')
        fig_reconstruction.savefig('images/reconstruction_%08d.png' % t_i)

        # %%
        # Plot manifold of latent layer
        zs = sess.run(ae['z'], feed_dict={ae['x']: xs})
        print("manifold obtained")
        ax_image_manifold.clear()
        ax_image_manifold.scatter(zs[:, 0], zs[:, 1],
                                  c=np.argmax(ys, 1), alpha=0.2)
        ax_image_manifold.set_xlim([-6, 6])
        ax_image_manifold.set_ylim([-6, 6])
        ax_image_manifold.axis('off')
        fig_image_manifold.savefig('images/image_manifold_%08d.png' % t_i)

        t_i += 1

        print('Train cost:', train_cost /
              (num_examples // batch_size))

        valid_cost = 0
        validation_examples = mnist.validation.num_examples

        for batch_i in range(validation_examples // batch_size):
            if mnist_flag:
                batch_xs, _ = mnist.validation.next_batch(batch_size)
            else:
                batch_xs, _ = get_normalized_x_y(val)
            valid_cost += sess.run([ae['cost']],
                                   feed_dict={ae['x']: batch_xs})[0]
        print('Validation cost:', valid_cost /
              (validation_examples // batch_size))


def montage_batch(images):
    """Draws all filters (n_input * n_output filters) as a
    montage image separated by 1 pixel borders.
    Parameters
    ----------
    batch : numpy.ndarray
        Input array to create montage of.
    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    m = np.ones(
        (images.shape[1] * n_plots + n_plots + 1,
         images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter, ...]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                1 + j + j * img_w:1 + j + (j + 1) * img_w, :] = this_img
    return m


def get_normalized_x_y(validation):
    if len(validation) == 0:
        return [], []
    validation_mean_x = np.mean([a[0] for a in validation])
    validation_mean_y = np.mean([a[1] for a in validation])
    validation_set = np.array(validation)
    validation_sources = [a[0] for a in validation_set]
    validation_targets = [a[1] for a in validation_set]

    validation_xs = np.zeros(
        [len(validation_sources), validation_sources[0].shape[0], validation_sources[0].shape[1], 1])
    for j, a in enumerate(validation_sources):
        validation_xs[j][:, :, 0] = a
    validation_ys = np.zeros(
        [len(validation_targets), validation_targets[0].shape[0], validation_targets[0].shape[1], 1])
    for j, a in enumerate(validation_targets):
        validation_ys[j][:, :, 0] = a
    validation_xs_norm = np.array([img - validation_mean_x for img in validation_xs])
    validation_ys_norm = np.array([img - validation_mean_y for img in validation_ys])
    return validation_xs_norm, validation_ys_norm


def train_autoencoder(ae, sess, train, validation, test, batch_size, n_epochs, learning_rate=0.01):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    train_mean_x = np.mean([a[0] for a in train])
    train_mean_y = np.mean([a[1] for a in train])

    training_set = np.array(train)
    validation_xs_norm, validation_ys_norm = get_normalized_x_y(validation)
    test_xs_norm, test_ys_norm = get_normalized_x_y(test)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)
    for epoch_i in range(n_epochs):
        perms = np.random.permutation(training_set)
        for i in range(len(training_set) / batch_size):
            batch = perms[i * batch_size:(i + 1) * batch_size, :]
            sources = [a[0] for a in batch]
            targets = [a[1] for a in batch]

            # TODO reformat
            batch_xs = np.zeros([len(sources), sources[0].shape[0], sources[0].shape[1], 1])
            for j, a in enumerate(sources):
                batch_xs[j][:, :, 0] = a
            batch_ys = np.zeros([len(targets), targets[0].shape[0], targets[0].shape[1], 1])
            for j, a in enumerate(targets):
                batch_ys[j][:, :, 0] = a

            train_source = np.array([img - train_mean_x for img in batch_xs])
            train_target = np.array([img - train_mean_y for img in batch_ys])

            sess.run(optimizer, feed_dict={ae['x']: train_source,
                                           ae['target']: train_target})

        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train_source, ae['target']: train_target}))

        # print("INSPECTING")
        # print(epoch_i, sess.run(ae['inspect'], feed_dict={ae['x']: train_source, ae['target']: train_target}))
        # print("++++")
        # print('y', sess.run(ae['y'], feed_dict={ae['x']: train_source, ae['target']: train_target}).shape)
        # print('z_sigma', sess.run(ae['z_sigma'], feed_dict={ae['x']: train_source, ae['target']: train_target}))
        # print('z_mean', sess.run(ae['z_mean'], feed_dict={ae['x']: train_source, ae['target']: train_target}))

        if epoch_i % 50 == 0 and len(validation) > 0:
            print("Validation", sess.run(ae['cost'], feed_dict={ae['x']: validation_xs_norm,
                                                                ae['target']: validation_ys_norm}))
    if len(test) > 0:
        print("Test", sess.run(ae['cost'], feed_dict={ae['x']: test_xs_norm,
                                                      ae['target']: test_ys_norm}))
    save_path = saver.save(sess, "./AE.ckpt")
    return ae


def split_dataset(dataset, test_split=.1, validation_split=.1):
    dataset = np.random.permutation(dataset)
    train_split = 1 - test_split - validation_split
    train_limit = int(train_split * len(dataset))
    validation_limit = int((train_split + test_split) * len(dataset))
    return dataset[:train_limit], dataset[train_limit: validation_limit], dataset[validation_limit:]


def vanilla_autoencoder(test_split=.1, validation_split=.1, autoencode=False):
    data_and_path, fs = get_male_female_pairs('encoder_data/DAPS/small_test/cut', product=False, subsample=-1)
    print("data loaded")
    print("Working with %s examples" % len(data_and_path))
    input_data = [a[0] for a in data_and_path]
    t_dim = input_data[0].shape[0]
    f_dim = input_data[0].shape[1]
    ae, shapes = VAE(input_shape=[None, t_dim, f_dim, 1],
                     n_filters=[1, 3, 3, 3],
                     filter_sizes=[4, 4, 4, 4], z_dim=50)
    sess = tf.Session()
    train, val, test = split_dataset(data_and_path, test_split, validation_split)
    if autoencode:
        train_model = [[t[1], t[1], t[-1]] for t in train]
        val_model = [[t[1], t[1], t[-1]] for t in val]
        test_model = [[t[1], t[1], t[-1]] for t in test]
        ae = train_autoencoder(ae, sess, train_model, val_model, test_model, batch_size=10, n_epochs=500)
    else:
        ae = train_autoencoder(ae, sess, train, val, test, batch_size=10, n_epochs=10)
    print(len(train), len(val), len(test), len(data_and_path))
    # plot_spectrograms(train, sess, ae, t_dim, f_dim)
    output_examples(train, ae, sess, fs, 'ae/train')
    output_examples(train, ae, sess, fs, 'ae/train', sources_index=1)
    output_examples(val, ae, sess, fs, 'ae/val')
    output_examples(test, ae, sess, fs, 'ae/test')
    plt.show()


def output_examples(data, model, sess, fs, folder, sources_index=0):
    source = [d[sources_index] for d in data]
    test_xs = np.zeros([len(source), source[0].shape[0], source[0].shape[1], 1])
    for i, a in enumerate(source):
        test_xs[i][:, :, 0] = a
    mean_img = np.mean(test_xs)
    test_xs_norm = np.array([img - mean_img for img in test_xs])

    recon = sess.run(model['y'], feed_dict={model['x']: test_xs_norm})
    for i in range(len(data)):
        output_filename = fft_to_audio('encoder_data/outputs/%s/a-%s_%s_%s' % (folder, i, sources_index, data[i][-1]),
                                       recon[i, ..., 0].T, fs, entire_path=True)


def plot_spectrograms(data, sess, ae, t_dim, f_dim):
    n_examples = 2

    source = [d[0] for d in data]
    target = [d[1] for d in data]
    test_xs = np.zeros([len(source), source[0].shape[0], source[0].shape[1], 1])
    for i, a in enumerate(source):
        test_xs[i][:, :, 0] = a
    mean_img = np.mean(test_xs)
    test_xs_norm = np.array([img - mean_img for img in test_xs])

    test_ys = np.zeros([len(target), target[0].shape[0], target[0].shape[1], 1])
    for j, a in enumerate(target):
        test_ys[j][:, :, 0] = a

    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
    print(recon.shape)
    print "z"

    fig, axs = plt.subplots(3, n_examples, figsize=(n_examples, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (t_dim, f_dim)), aspect='auto')
        axs[1][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (t_dim * f_dim,)) + mean_img,
                (t_dim, f_dim)), aspect='auto')
        axs[2][example_i].imshow(
            np.reshape(test_ys[example_i, :], (t_dim, f_dim)), aspect='auto')
        print(recon[example_i, ..., 0].T.shape)


# %%
if __name__ == '__main__':
    # test_mnist()
    # test_data()
    vanilla_autoencoder(autoencode=True)
    # test(mnist_flag=False)
