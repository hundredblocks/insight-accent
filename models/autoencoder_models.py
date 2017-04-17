import math
import logging

import numpy as np
import tensorflow as tf

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def cross_autoencoder(input_shape=[None, 784],
                      n_filters=[1, 10, 10, 10],
                      filter_sizes=[3, 3, 3, 3]):
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')

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

    # Build the encoder
    encoder = []
    shapes = []
    outputs = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        logging.info(layer_i)
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

    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()

    outputs.reverse()

    # Build the decoder
    for layer_i, shape in enumerate(shapes):
        W_out = encoder[layer_i]
        b = tf.Variable(tf.zeros([W_out.get_shape().as_list()[2]]))

        deconv = tf.add(
            tf.nn.conv2d_transpose(
                current_input, W_out,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b)
        logging.info(deconv.get_shape())

        output = tf.nn.relu(deconv)
        current_input = output

    # now have the reconstruction through the network
    y = current_input
    target = tf.placeholder(tf.float32, input_shape, name='target')
    # cost function measures pixel-wise difference
    cost = tf.reduce_mean(tf.square(y - target))

    # %%
    return {'x': x, 'z': z, 'y': y, 'target': target, 'cost': cost}, shapes


def vae(input_shape=[None, 784],
        n_filters=[1, 10, 10, 10],
        filter_sizes=[3, 3, 3, 3],
        z_dim=50, loss_function='l2', encode_with_latent=False, learning_rate=0.001):
    x = tf.placeholder(tf.float32, input_shape, name='x')
    dropout = tf.placeholder_with_default(1., shape=[], name="dropout")
    dropout_fc = tf.placeholder_with_default(1., shape=[], name="dropout")

    if encode_with_latent:
        class_vector = tf.placeholder(tf.float32, shape=[None, 2], name='class_vector')

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
    logging.info("input tensor %s" % current_input)

    encoder = []
    shapes = []
    outputs = []
    w_dims = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        w_dims.append(n_input)
        shapes.append(current_input.get_shape().as_list())

        outputs.append(current_input)
        conv_pre_norm = conv_layer(current_input, [filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_output],
                                   -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input), n_output)
        conv = tf.contrib.layers.batch_norm(conv_pre_norm)
        output_pre_drop = tf.nn.relu(conv)
        output = tf.nn.dropout(output_pre_drop, dropout)
        logging.info("layer shape %s" % output.get_shape())

        current_input = output

    _, w, h, c = output.get_shape().as_list()
    output_flatten = tf.reshape(output, [-1, w * h * c])
    output_shape = output_flatten.get_shape().as_list()[-1]
    logging.info("flattened shape %s"% output_flatten.get_shape())

    z_mean_pre_drop = fc_layer(output_flatten, output_flatten.get_shape().as_list()[1], z_dim, output_shape)
    z_mean = tf.nn.dropout(z_mean_pre_drop, dropout_fc)

    z_sigma_pre_drop = fc_layer(output_flatten, output_flatten.get_shape().as_list()[1], z_dim, output_shape)
    z_sigma = tf.nn.dropout(z_sigma_pre_drop, dropout_fc)
    logging.info("z_mean %s" % z_mean.get_shape())

    z = sample_gaussian(z_mean, z_sigma)
    logging.info("z %s" % z.get_shape())

    if encode_with_latent:
        z_class = tf.concat([z, class_vector], 1)
        logging.info("encoded class in vector %s" % z_class.get_shape())
        # TODO generalize
        rec_pre_drop = fc_layer(z_class, z_dim + 2, output_flatten.get_shape().as_list()[1], output_shape)
    else:
        rec_pre_drop = fc_layer(z, z_dim, output_flatten.get_shape().as_list()[1], output_shape)
    rec = tf.nn.dropout(rec_pre_drop, dropout_fc)
    logging.info("decoder input flat shape %s" % rec.get_shape())

    batch = tf.shape(output)[0]
    target = [batch]
    target.extend((output.get_shape().as_list()[1:]))

    current_input = tf.reshape(rec, target)

    logging.info("input to decoder %s" % current_input.get_shape())

    encoder.reverse()
    shapes.reverse()

    outputs.reverse()

    for layer_i, shape in enumerate(shapes):
        n_output = w_dims[-(layer_i + 1)]
        n_input = current_input.get_shape().as_list()[3]
        weights_shape = [filter_sizes[-layer_i + 1], filter_sizes[-layer_i + 1], n_output, n_input]

        filter_shape = [tf.shape(x)[0], shape[1], shape[2], shape[3]]
        logging.info("filter shape %s" % filter_shape)

        deconv = deconv_layer(current_input, weights_shape, filter_shape,
                              -1.0 / math.sqrt(n_output), 1.0 / math.sqrt(n_output), n_output)

        if layer_i != len(shapes) - 1:
            output_pre_drop = tf.nn.relu(deconv)
        else:
            output_pre_drop = deconv
        output = tf.nn.dropout(output_pre_drop, dropout)
        current_input = output
        logging.info(output.get_shape())

    y = current_input
    logging.info(y.get_shape())

    if loss_function == 'l2':
        logging.info('l2 loss function chosen')
        rec_cost = l2_loss(y, x_tensor)
    elif loss_function == 'l1':
        logging.info('l1 loss function chosen')
        rec_cost = l1_loss(y, x_tensor)
    else:
        logging.info('cross entropy loss function chosen')
        rec_cost = cross_entropy(y, x_tensor)
    ampl_factor = 1
    vae_loss_kl = ampl_factor * kl_div(z_mean, z_sigma)

    logging.info("vae loss %s" % vae_loss_kl.get_shape())
    logging.info("rec loss %s" % rec_cost.get_shape())
    cost = tf.reduce_mean(rec_cost + vae_loss_kl)

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    trainable = tf.trainable_variables()
    grads_and_vars = optimizer.compute_gradients(cost, trainable)
    grads_and_vars = [g for g in grads_and_vars if g[0] is not None]
    clipped = [(tf.clip_by_value(grad, -1., 1.), tvar) for grad, tvar in grads_and_vars]

    train_op = optimizer.apply_gradients(clipped, global_step=global_step, name="minimize_cost")
    param_dict = {'x': x, 'z': z, 'y': y, 'cost': cost, 'rec_cost': rec_cost, 'vae_loss_kl': vae_loss_kl,
                  'z_mean': z_mean, 'z_sigma': z_sigma,
                  'train_op': train_op, 'dropout': dropout, 'dropout_fc': dropout_fc}
    if encode_with_latent:
        param_dict['class'] = class_vector
    return param_dict, shapes


def sample_gaussian(mu, log_sigma):
    epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
    return mu + epsilon * tf.exp(log_sigma)  # N(mu, I * sigma**2)


def bias_variable(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)


def weight_variable(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)


def fc_weight_variable(in_dim, out_dim, output_shape):
    return tf.Variable(
        tf.random_uniform([in_dim, out_dim], -1.0 / math.sqrt(output_shape), 1.0 / math.sqrt(output_shape)))


def conv_layer(prev_layer, shape, min_val, max_val, n_output):
    weights = tf.Variable(tf.random_uniform(shape, min_val, max_val))
    b = tf.Variable(tf.zeros([n_output]))
    return tf.add(tf.nn.conv2d(prev_layer, weights, strides=[1, 2, 2, 1], padding='SAME'), b)


def deconv_layer(prev_layer, shape, filter_shape, min_val, max_val, n_output):
    weights = tf.Variable(tf.random_uniform(shape, min_val, max_val))
    b = tf.Variable(tf.zeros([n_output]))
    conv_2d = tf.nn.conv2d_transpose(
        value=prev_layer, filter=weights,
        output_shape=tf.stack(filter_shape),
        strides=[1, 2, 2, 1], padding='SAME')
    return tf.add(conv_2d, b)


def fc_layer(prev_layer, in_dim, out_dim, output_shape):
    w_m = fc_weight_variable(in_dim, out_dim, output_shape)
    b_m = tf.Variable(tf.zeros([out_dim]))
    return tf.matmul(prev_layer, w_m) + b_m


def cross_entropy(y, target, offset=1e-7):
    obs_ = tf.clip_by_value(y, offset, 5 - offset)
    return -tf.reduce_mean(target * tf.log(obs_) + (1 - target) * tf.log(1 - obs_), [1, 2])


def l2_loss(y, target, offset=1e-7):
    obs_ = tf.clip_by_value(y, offset, 5 - offset)
    return tf.reduce_mean(tf.square(obs_ - target), [1, 2])


def l1_loss(y, target, offset=1e-7):
    obs_ = tf.clip_by_value(y, offset, 5 - offset)
    return tf.reduce_mean(tf.abs(obs_ - target), [1, 2])


def kl_div(z_mean, z_sigma, offset=1e-7):
    z_sigma_ = tf.clip_by_value(z_sigma, offset, 5 - offset)
    z_mean_ = tf.clip_by_value(z_mean, offset, 5 - offset)
    vae_loss_kl = -0.5 * tf.reduce_mean(1 + z_sigma_ - tf.square(z_mean_) - tf.exp(z_sigma_), 1)
    return vae_loss_kl
