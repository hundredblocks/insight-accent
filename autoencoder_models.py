"""from https://github.com/pkmital/tensorflow_tutorials/blob/master/python/09_convolutional_autoencoder.py
Tutorial on how to create a convolutional autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
import tensorflow as tf
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_fetch import preprocess_and_load, get_all_audio_in_folder, get_male_female_pairs
from utils import fft_to_audio


def cross_autoencoder(input_shape=[None, 784],
                      n_filters=[1, 10, 10, 10],
                      filter_sizes=[3, 3, 3, 3],
                      corruption=False):
    """Build a deep denoising autoencoder w/ tied weights.

    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """
    # %%
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')

    # %%
    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions: root is %s' % x_dim)
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    # %%
    # Optionally apply denoising autoencoder
    # if corruption:
    #     current_input = corrupt(current_input)

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    outputs = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        print(layer_i)
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())

        outputs.append(current_input)
        W_in = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))

        W_out = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W_out)
        conv = tf.add(tf.nn.conv2d(
            current_input, W_in, strides=[1, 2, 2, 1], padding='SAME'), b)
        output = tf.nn.relu(conv)

        current_input = output

    # %%
    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()

    outputs.reverse()

    # %%
    # Build the decoder
    for layer_i, shape in enumerate(shapes):
        # TODO Different weights
        W_out = encoder[layer_i]
        b = tf.Variable(tf.zeros([W_out.get_shape().as_list()[2]]))

        deconv = tf.add(
            tf.nn.conv2d_transpose(
                current_input, W_out,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b)

        # output_res = tf.add(deconv, outputs[layer_i])
        # output = tf.nn.relu(output_res)

        output = tf.nn.relu(deconv)
        current_input = output

    # %%
    # now have the reconstruction through the network
    y = current_input
    target = tf.placeholder(tf.float32, input_shape, name='target')
    # cost function measures pixel-wise difference
    cost = tf.reduce_mean(tf.square(y - target))

    # %%
    return {'x': x, 'z': z, 'y': y, 'target': target, 'cost': cost}, shapes


def VAE(input_shape=[None, 784],
        n_components_encoder=2048,
        n_components_decoder=2048,
        n_hidden=2,
        debug=False):
    # %%
    # Input placeholder
    if debug:
        input_shape = [50, 784]
        x = tf.Variable(np.zeros((input_shape), dtype=np.float32))
    else:
        x = tf.placeholder(tf.float32, input_shape)

    activation = tf.nn.softplus

    dims = x.get_shape().as_list()
    n_features = dims[1]

    W_enc1 = weight_variable([n_features, n_components_encoder])
    b_enc1 = bias_variable([n_components_encoder])
    h_enc1 = activation(tf.matmul(x, W_enc1) + b_enc1)

    W_enc2 = weight_variable([n_components_encoder, n_components_encoder])
    b_enc2 = bias_variable([n_components_encoder])
    h_enc2 = activation(tf.matmul(h_enc1, W_enc2) + b_enc2)

    W_enc3 = weight_variable([n_components_encoder, n_components_encoder])
    b_enc3 = bias_variable([n_components_encoder])
    h_enc3 = activation(tf.matmul(h_enc2, W_enc3) + b_enc3)

    W_mu = weight_variable([n_components_encoder, n_hidden])
    b_mu = bias_variable([n_hidden])

    W_log_sigma = weight_variable([n_components_encoder, n_hidden])
    b_log_sigma = bias_variable([n_hidden])

    z_mu = tf.matmul(h_enc3, W_mu) + b_mu
    z_log_sigma = 0.5 * (tf.matmul(h_enc3, W_log_sigma) + b_log_sigma)

    # %%
    # Sample from noise distribution p(eps) ~ N(0, 1)
    if debug:
        epsilon = tf.random_normal(
            [dims[0], n_hidden])
    else:
        epsilon = tf.random_normal(
            tf.stack([tf.shape(x)[0], n_hidden]))

    # Sample from posterior
    z = z_mu + tf.exp(z_log_sigma) * epsilon

    W_dec1 = weight_variable([n_hidden, n_components_decoder])
    b_dec1 = bias_variable([n_components_decoder])
    h_dec1 = activation(tf.matmul(z, W_dec1) + b_dec1)

    W_dec2 = weight_variable([n_components_decoder, n_components_decoder])
    b_dec2 = bias_variable([n_components_decoder])
    h_dec2 = activation(tf.matmul(h_dec1, W_dec2) + b_dec2)

    W_dec3 = weight_variable([n_components_decoder, n_components_decoder])
    b_dec3 = bias_variable([n_components_decoder])
    h_dec3 = activation(tf.matmul(h_dec2, W_dec3) + b_dec3)

    W_mu_dec = weight_variable([n_components_decoder, n_features])
    b_mu_dec = bias_variable([n_features])
    y = tf.nn.sigmoid(tf.matmul(h_dec3, W_mu_dec) + b_mu_dec)

    # p(x|z)
    log_px_given_z = -tf.reduce_sum(
        x * tf.log(y + 1e-10) +
        (1 - x) * tf.log(1 - y + 1e-10), 1)

    # d_kl(q(z|x)||p(z))
    # Appendix B: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * tf.reduce_sum(
        1.0 + 2.0 * z_log_sigma - tf.square(z_mu) - tf.exp(2.0 * z_log_sigma),
        1)
    loss = tf.reduce_mean(log_px_given_z + kl_div)

    return {'cost': loss, 'x': x, 'z': z, 'y': y}


def test_mnist():
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
    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    ae = VAE()

    # %%
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # %%
    # Fit all training data
    t_i = 0
    batch_size = 100
    n_epochs = 10
    n_examples = 20
    test_xs, _ = mnist.test.next_batch(n_examples)
    xs, ys = mnist.test.images, mnist.test.labels
    fig_manifold, ax_manifold = plt.subplots(1, 1)
    fig_reconstruction, axs_reconstruction = plt.subplots(2, n_examples, figsize=(10, 2))
    fig_image_manifold, ax_image_manifold = plt.subplots(1, 1)
    for epoch_i in range(n_epochs):
        print('--- Epoch', epoch_i)
        train_cost = 0
        print("Number train examples %s" % mnist.train.num_examples)
        print("Num batches %s" % (mnist.train.num_examples // batch_size))
        for batch_i in range(mnist.train.num_examples // batch_size):
            # print(batch_i)
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train_cost += sess.run([ae['cost'], optimizer],
                                   feed_dict={ae['x']: batch_xs})[0]
        # %%
        # Plot example reconstructions from latent layer
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
              (mnist.train.num_examples // batch_size))

        valid_cost = 0
        for batch_i in range(mnist.validation.num_examples // batch_size):
            batch_xs, _ = mnist.validation.next_batch(batch_size)
            valid_cost += sess.run([ae['cost']],
                                   feed_dict={ae['x']: batch_xs})[0]
        print('Validation cost:', valid_cost /
              (mnist.validation.num_examples // batch_size))


def weight_variable(shape):
    '''Helper function to create a weight variable initialized with
    a normal distribution
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    '''Helper function to create a bias variable initialized with
    a constant value.
    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)


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
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train_source,
                                                       ae['target']: train_target}))
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


def vanilla_autoencoder(test_split=.1, validation_split=.1):
    data_and_path, fs = get_male_female_pairs('encoder_data/DAPS/small_test/cut', product=False)
    print("data loaded")
    # data_and_path = data_and_path[:2]
    print("Working with %s examples" % len(data_and_path))
    input_data = [a[0] for a in data_and_path]
    t_dim = input_data[0].shape[0]
    f_dim = input_data[0].shape[1]
    ae, shapes = cross_autoencoder(input_shape=[None, t_dim, f_dim, 1],
                                   n_filters=[1, 10, 10, 10],
                                   filter_sizes=[4, 4, 4, 4], )
    sess = tf.Session()
    train, val, test = split_dataset(data_and_path, test_split, validation_split)
    # train, val, test = data_and_path, [], []
    print(len(train), len(val), len(test), len(data_and_path))
    ae = train_autoencoder(ae, sess, train, val, test, batch_size=1, n_epochs=10)

    # plot_spectrograms(train, sess, ae, t_dim, f_dim)
    output_examples(train, ae, sess, fs, 'ae/train')
    output_examples(val, ae, sess, fs, 'ae/val')
    output_examples(test, ae, sess, fs, 'ae/test')
    plt.show()


def output_examples(data, model, sess, fs, folder):
    source = [d[0] for d in data]
    test_xs = np.zeros([len(source), source[0].shape[0], source[0].shape[1], 1])
    for i, a in enumerate(source):
        test_xs[i][:, :, 0] = a
    mean_img = np.mean(test_xs)
    test_xs_norm = np.array([img - mean_img for img in test_xs])

    recon = sess.run(model['y'], feed_dict={model['x']: test_xs_norm})
    for i in range(len(data)):
        output_filename = fft_to_audio('encoder_data/outputs/%s/a-%s_%s' % (folder, i, data[i][-1]),
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

    # print(test_xs_norm.shape)
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
    # vanilla_autoencoder()
    test_mnist()
