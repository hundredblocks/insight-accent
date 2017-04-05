import matplotlib
import os
import numpy as np
import tensorflow as tf

from autoencoder_models import vae, VAE_MNIST, cross_autoencoder

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_fetch import get_male_female_pairs, get_all_autoencoder_audio_in_folder
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
        ae = vae(input_shape=[None, tdim * fdim])

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

            curr_cost, kl, dist, _ = sess.run([ae['cost'], ae['kl_div'], ae['log_px_given_z'], optimizer],
                                              feed_dict={ae['x']: batch_xs})
            train_cost += curr_cost
            print(curr_cost, np.mean(kl), np.sum(kl), np.mean(dist), np.sum(dist))
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

    validation_xs = np.zeros([len(validation_sources), validation_sources[0].shape[0], validation_sources[0].shape[1], 1])
    for j, a in enumerate(validation_sources):
        validation_xs[j][:, :, 0] = a
    validation_ys = np.zeros([len(validation_targets), validation_targets[0].shape[0], validation_targets[0].shape[1], 1])
    for j, a in enumerate(validation_targets):
        validation_ys[j][:, :, 0] = a
    validation_xs_norm = np.array([img - validation_mean_x for img in validation_xs])
    validation_ys_norm = np.array([img - validation_mean_y for img in validation_ys])
    return validation_xs_norm, validation_ys_norm


def get_normalized(data, data_mean=None, data_std=None):
    np_data_only = np.array([a[0] for a in data])
    if len(data) == 0:
        return [], []
    if data_mean is None:
        data_mean = np.mean(np_data_only)
        print("MEAN", data_mean)
    if data_std is None:
        data_std = np.std(np_data_only)
        print("STD", data_std)

    np_normalized = (np_data_only - data_mean) / data_std

    tensor = np.zeros([len(np_data_only), np_data_only[0].shape[0], np_data_only[0].shape[1], 1])
    for j, a in enumerate(np_normalized):
        tensor[j][:, :, 0] = a
    tensor_with_label = [np.array([tensor[i, :, :], data[i][1], data[i][-1]]) for i in range(len(data))]
    # validation_normalized = np.array([img - data_mean for img in validation_xs])
    return tensor_with_label, data_mean, data_std


def train_autoencoder(ae, sess, train_norm, validation_norm, test_norm, batch_size,
                      n_epochs, encode_with_latent=False, learning_rate=0.01):
    sess.run(tf.global_variables_initializer())
    keep_prob = 0.9
    keep_prob_fc = 0.7
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)
    for epoch_i in range(n_epochs):
        perms = np.random.permutation(train_norm)
        for i in range(len(train_norm) / batch_size):
            batch = perms[i * batch_size:(i + 1) * batch_size, :]
            batch_input = [b[0] for b in batch]
            batch_class = [b[1] for b in batch]
            feed_dict_train = {
                ae['x']: batch_input,
                ae['dropout']: keep_prob,
                ae['dropout_fc']: keep_prob_fc
            }
            if encode_with_latent:
                feed_dict_train[ae['class']] = batch_class

            sess.run(ae['train_op'],
                     feed_dict=feed_dict_train)

        feed_dict_eval = {ae['x']: [a[0] for a in train_norm]}
        if encode_with_latent:
            feed_dict_eval[ae['class']] = [a[1] for a in train_norm]
        cost, rec_cost, kl_cost = sess.run([ae['cost'], ae['rec_cost'], ae['vae_loss_kl']],
                                           feed_dict=feed_dict_eval)

        print(epoch_i, cost, np.mean(rec_cost), np.mean(kl_cost))

        if epoch_i % 10 == 0 and len(validation_norm) > 0:
            feed_dict_val = {ae['x']: [a[0] for a in validation_norm]}
            if encode_with_latent:
                feed_dict_val[ae['class']] = [a[1] for a in validation_norm]
            print("Validation", sess.run(ae['cost'], feed_dict=feed_dict_val))
        if epoch_i % 50 == 0:
            save_path = saver.save(sess, "./AE_%s.ckpt" % epoch_i)
    if len(test_norm) > 0:
        feed_dict_test = {ae['x']: [a[0] for a in test_norm]}
        if encode_with_latent:
            feed_dict_test[ae['class']] = [a[1] for a in test_norm]
        print("Test", sess.run(ae['cost'], feed_dict=feed_dict_test))
    save_path = saver.save(sess, "./AE_final.ckpt")
    return ae


def train_crossautoencoder(ae, sess, train, validation, test, batch_size, n_epochs, learning_rate=0.01):
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    train_mean_x = np.mean([a[0] for a in train])
    train_mean_y = np.mean([a[1] for a in train])

    training_set = np.array(train)
    train_xs_norm, train_ys_norm = get_normalized_x_y(train)
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
            # sess.run(optimizer, feed_dict={ae['x']: train_source,ae['target']: train_target})
            sess.run(ae['train_op'],
                     feed_dict={ae['x']: train_source, ae['target']: train_target})

        cost, rec_cost, kl_cost = sess.run([ae['cost'], ae['rec_cost'], ae['vae_loss_kl']],
                                           feed_dict={ae['x']: train_xs_norm, ae['target']: train_ys_norm})

        print(cost, np.mean(rec_cost), np.mean(kl_cost))
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train_xs_norm, ae['target']: train_ys_norm}))

        if epoch_i % 10 == 0 and len(validation) > 0:
            print("Validation", sess.run(ae['cost'], feed_dict={ae['x']: validation_xs_norm,
                                                                ae['target']: validation_ys_norm}))
    if len(test) > 0:
        print("Test", sess.run(ae['cost'], feed_dict={ae['x']: test_xs_norm,
                                                      ae['target']: test_ys_norm}))
    save_path = saver.save(sess, "./AE.ckpt")
    return ae


def split_dataset(dataset, test_split=.1, validation_split=.1, offset=None):
    if offset is None:
        offset = np.random.randint(0, len(dataset) - 1)
    train_split = 1 - test_split - validation_split

    train_size = int(train_split * len(dataset))
    val_size = int(validation_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_start = 0
    train_end = train_size
    val_end = train_end + val_size

    offset_arr = dataset[offset:] + dataset[:offset]
    return offset_arr[train_start:train_end], offset_arr[train_end:val_end], \
           offset_arr[val_end:val_end + test_size], offset


def vanilla_autoencoder(n_filters=None, filter_sizes=None, learning_rate=0.001,
                        z_dim=50, subsample=-1, batch_size=10,
                        n_epochs=100, loss_function='l2', test_split_ratio=.1,
                        val_split_ratio=.1, autoencode=False,
                        data_path='encoder_data/DAPS/small_test/cut', encode_with_latent=False):
    if not n_filters:
        n_filters = [1, 3, 3, 3]
    if not filter_sizes:
        filter_sizes = [4, 4, 4, 4]
    data_and_path_female, fs = get_all_autoencoder_audio_in_folder(os.path.join(data_path, 'female'),
                                                                   subsample=subsample, class_label=[1, 0])
    data_and_path_male, fs = get_all_autoencoder_audio_in_folder(os.path.join(data_path, 'male'),
                                                                 subsample=subsample, class_label=[0, 1])
    print("data loaded")
    input_data = [a[0] for a in data_and_path_female]
    t_dim = input_data[0].shape[0]
    f_dim = input_data[0].shape[1]
    ae, shapes = vae(input_shape=[None, t_dim, f_dim, 1],
                     n_filters=n_filters,
                     filter_sizes=filter_sizes, z_dim=z_dim,
                     loss_function=loss_function, encode_with_latent=encode_with_latent, learning_rate=learning_rate)

    sess = tf.Session()
    train_split, val_split, test_split, offset = split_dataset(data_and_path_female, test_split_ratio,
                                                               val_split_ratio)

    if encode_with_latent:
        train_split_m, val_split_m, test_split_m, _ = split_dataset(data_and_path_male, test_split_ratio,
                                                                    val_split_ratio, offset=offset)
        train_split.extend(train_split_m)
        val_split.extend(val_split_m)
        test_split.extend(test_split_m)

    print(len(train_split), len(val_split), len(test_split))
    train_norm, data_mean, data_var = get_normalized(train_split)
    val_norm, _, _ = get_normalized(val_split, data_mean, data_var)
    test_norm, _, _ = get_normalized(test_split, data_mean, data_var)

    print("Total %s. Training on %s, validating on %s, testing on %s" % (
        len(data_and_path_female), len(train_split), len(val_split), len(test_split)))
    if autoencode:
        ae = train_autoencoder(ae, sess, train_norm, val_norm, test_norm, batch_size=batch_size,
                               n_epochs=n_epochs, encode_with_latent=encode_with_latent)
    else:
        ae = train_crossautoencoder(ae, sess, train_split, val_split, test_split, batch_size=batch_size,
                                    n_epochs=n_epochs)

    # plot_spectrograms(train, sess, ae, t_dim, f_dim)

    to_plot_train = np.random.permutation(train_split)[:min(len(train_split), 15)]
    to_plot_val = np.random.permutation(val_split)[:min(len(val_split), 15)]
    to_plot_test = np.random.permutation(test_split)[:min(len(test_split), 15)]

    output_examples(to_plot_train, ae, sess, fs, 'ae/train',
                    data_mean, data_var, encode_with_latent=encode_with_latent, class_label=[1, 0])

    output_examples(to_plot_val, ae, sess, fs, 'ae/val',
                    data_mean, data_var, encode_with_latent=encode_with_latent, class_label=[1, 0])

    output_examples(to_plot_test, ae, sess, fs, 'ae/test',
                    data_mean, data_var, encode_with_latent=encode_with_latent, class_label=[1, 0])
    output_examples(data_and_path_male, ae, sess, fs, 'ae/male',
                    data_mean, data_var, encode_with_latent=encode_with_latent, class_label=[1, 0])
    output_examples(data_and_path_male, ae, sess, fs, 'ae/male_morph',
                    data_mean, data_var, encode_with_latent=encode_with_latent, class_label=[1, 0])

    plt.show()


def output_examples(data, model, sess, fs, folder, data_mean, data_std, encode_with_latent=False, class_label=None):
    source_norm, _, _ = get_normalized(data, data_mean, data_std)

    feed_dict = {
        model['x']: [a[0] for a in source_norm],
    }
    if encode_with_latent:
        if class_label is None:
            class_label = [1, 0]
        feed_dict[model['class']] = [class_label for a in source_norm]

    recon = sess.run(model['y'], feed_dict=feed_dict)
    for i in range(len(data)):
        data_i = recon[i, ..., 0].T
        data_norm = data_i * data_std + data_mean
        output_filename = fft_to_audio('encoder_data/outputs/%s/unnorm-%s_%s' % (folder, i, data[i][-1]),
                                       data_norm, fs, entire_path=True)


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
    vanilla_autoencoder(n_filters=[1, 4, 6, 8], filter_sizes=[3, 3, 3, 3], learning_rate=0.001,
                        # z_dim=50, subsample=20, batch_size=4, n_epochs=600,
                        z_dim=50, subsample=10, batch_size=2, n_epochs=1000,
                        loss_function='l2', autoencode=True, data_path='encoder_data/DAPS/f3_m4/cut_1000_step_100',
                        encode_with_latent=True)
