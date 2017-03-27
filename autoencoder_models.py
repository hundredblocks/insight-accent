import math

import numpy as np
import tensorflow as tf


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
        print(deconv.get_shape())

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
        n_filters=[1, 10, 10, 10],
        filter_sizes=[3, 3, 3, 3],
        z_dim=50, loss_function='l2'):
    # %%
    # Input placeholder
    """Build a deep denoising autoencoder w/ tied weights.

        Parameters
        ----------
        input_shape : list, optional
            Description
        n_filters : list, optional
            Description
        filter_sizes : list, optional
            Description
        z_dim: number of dimensions of latent space
        loss_function: l2 or cross entropy

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
    print("current_input", current_input)
    n_points = x_tensor.get_shape().as_list()[1] * x_tensor.get_shape().as_list()[2]

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    outputs = []
    w_dims = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        w_dims.append(n_input)
        shapes.append(current_input.get_shape().as_list())

        outputs.append(current_input)

        output = tf.nn.relu(conv_layer(current_input, [filter_sizes[layer_i], filter_sizes[layer_i], n_input, n_output],
                                       -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input), n_output))

        print("W at layer", layer_i, [
            filter_sizes[layer_i],
            filter_sizes[layer_i],
            n_input, n_output])

        print(output.get_shape())

        current_input = output

    # TODO Temp
    _, w, h, c = output.get_shape().as_list()
    output_flatten = tf.reshape(output, [-1, w * h * c])
    output_shape = output_flatten.get_shape().as_list()[-1]
    print("output_flatten")
    print(output_flatten.get_shape())

    z_mean = fc_layer(output_flatten, output_flatten.get_shape().as_list()[1], z_dim, output_shape)
    z_sigma = fc_layer(output_flatten, output_flatten.get_shape().as_list()[1], z_dim, output_shape)
    print("z_mean")
    print(z_mean.get_shape())

    z = sample_gaussian(z_mean, z_sigma)
    print("z")
    print(z.get_shape())

    rec = fc_layer(z, z_dim, output_flatten.get_shape().as_list()[1], output_shape)
    print("reconstruction")
    print(rec.get_shape())

    batch = tf.shape(output)[0]
    target = [batch]
    target.extend((output.get_shape().as_list()[1:]))

    current_input = tf.reshape(rec, target)

    print("input to decoder")
    print(current_input.get_shape())

    encoder.reverse()
    shapes.reverse()

    outputs.reverse()

    # %%
    # Build the decoder
    for layer_i, shape in enumerate(shapes):
        n_output = w_dims[-(layer_i + 1)]
        n_input = current_input.get_shape().as_list()[3]
        weights_shape = [filter_sizes[-layer_i + 1], filter_sizes[-layer_i + 1], n_output, n_input]

        # [batch,

        filter_shape = [tf.shape(x)[0], shape[1], shape[2], shape[3]]
        print("filter shape")
        print(filter_shape)
        print(tf.stack(filter_shape))
        output = tf.nn.relu(
            deconv_layer(current_input, weights_shape, filter_shape,
                         -1.0 / math.sqrt(n_output), 1.0 / math.sqrt(n_output), n_output))

        current_input = output
        print(output.get_shape())

    # %%
    # now have the reconstruction through the network
    y = current_input
    print(y.get_shape())
    target = tf.placeholder(tf.float32, input_shape, name='target')
    # cost function measures pixel-wise difference

    if loss_function == 'l2':
        rec_cost = l2_loss(y, target)
    elif loss_function == 'l1':
        rec_cost = l1_loss(y, target)
    else:
        rec_cost = cross_entropy(y, target)
    ampl_factor = 1000
    vae_loss_kl = ampl_factor*kl_div(z_mean, z_sigma)
    # vae_loss_kl = tf.reduce_mean(vae_loss_kl) / n_points
    print("vae")
    print(vae_loss_kl.get_shape())
    print("rec")
    print(rec_cost.get_shape())
    # cost = tf.reduce_mean(vae_loss_kl)
    cost = tf.reduce_mean(rec_cost + vae_loss_kl)

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(0.01)
    trainable = tf.trainable_variables()
    grads_and_vars = optimizer.compute_gradients(cost, trainable)
    grads_and_vars = [g for g in grads_and_vars if g[0] is not None]
    clipped = [(tf.clip_by_value(grad, -1., 1.), tvar) for grad, tvar in grads_and_vars]

    train_op = optimizer.apply_gradients(clipped, global_step=global_step, name="minimize_cost")

    # %%
    return {'x': x, 'z': z, 'y': y, 'target': target, 'cost': cost, 'rec_cost': rec_cost, 'vae_loss_kl': vae_loss_kl,
            'z_mean': z_mean, 'z_sigma': z_sigma,
            'train_op': train_op}, shapes


def VAE_MNIST(input_shape=[None, 784],
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

    if len(x.get_shape()) == 2:
        # x_dim = np.sqrt(x.get_shape().as_list()[1])
        # if x_dim != int(x_dim):
        #     raise ValueError('Unsupported input dimensions: root is %s' % x_dim)
        # x_dim = int(x_dim)
        x_tensor = tf.reshape(x, [-1, input_shape[1]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    x = x_tensor

    activation = tf.nn.softplus

    dims = x.get_shape().as_list()
    n_features = dims[1]
    print("x")
    print(x.get_shape())
    W_enc1 = weight_variable([n_features, n_components_encoder])
    b_enc1 = bias_variable([n_components_encoder])
    h_enc1 = activation(tf.matmul(x, W_enc1) + b_enc1)

    print("h_enc1")
    print(h_enc1.get_shape())
    W_enc2 = weight_variable([n_components_encoder, n_components_encoder])
    b_enc2 = bias_variable([n_components_encoder])
    h_enc2 = activation(tf.matmul(h_enc1, W_enc2) + b_enc2)

    print("h_enc2")
    print(h_enc2.get_shape())
    W_enc3 = weight_variable([n_components_encoder, n_components_encoder])
    b_enc3 = bias_variable([n_components_encoder])
    h_enc3 = activation(tf.matmul(h_enc2, W_enc3) + b_enc3)

    print("h_enc3")
    print(h_enc3.get_shape())
    W_mu = weight_variable([n_components_encoder, n_hidden])
    b_mu = bias_variable([n_hidden])

    W_log_sigma = weight_variable([n_components_encoder, n_hidden])
    b_log_sigma = bias_variable([n_hidden])

    z_mu = tf.matmul(h_enc3, W_mu) + b_mu
    z_log_sigma = 0.5 * (tf.matmul(h_enc3, W_log_sigma) + b_log_sigma)

    print("z_mu")
    print(z_mu.get_shape())

    print("z_log_sigma")
    print(z_log_sigma.get_shape())
    # print(z_mu.get_shape())
    # print(z_log_sigma.get_shape())
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

    print("z")
    print(z.get_shape())

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

    print("y")
    print(y.get_shape())

    print("x")
    print(x.get_shape())
    # p(x|z)
    log_px_given_z = -tf.reduce_sum(
        x * tf.log(y + 1e-10) +
        (1 - x) * tf.log(1 - y + 1e-10), 1)

    print("ce_error")
    print(log_px_given_z.get_shape())

    # d_kl(q(z|x)||p(z))
    # Appendix B: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * tf.reduce_sum(
        1.0 + 2.0 * z_log_sigma - tf.square(z_mu) - tf.exp(2.0 * z_log_sigma),
        1)
    print("kl_div")
    print(kl_div.get_shape())

    loss = tf.reduce_mean(log_px_given_z + kl_div)
    print("loss")
    print(loss.get_shape())

    return {'cost': loss, 'kl_div': kl_div, 'log_px_given_z': log_px_given_z, 'x': x, 'z': z, 'y': y}


def sample_gaussian(mu, log_sigma):
    epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
    return mu + epsilon * tf.exp(log_sigma)  # N(mu, I * sigma**2)


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


def fc_weight_variable(in_dim, out_dim, output_shape):
    return tf.Variable(
        tf.random_uniform([
            in_dim, out_dim],
            -1.0 / math.sqrt(output_shape),
            1.0 / math.sqrt(output_shape)))


def conv_layer(prev_layer, shape, min_val, max_val, n_output):
    weights = tf.Variable(tf.random_uniform(shape, min_val, max_val))
    b = tf.Variable(tf.zeros([n_output]))

    return tf.add(tf.nn.conv2d(
        prev_layer, weights, strides=[1, 2, 2, 1], padding='SAME'), b)


def deconv_layer(prev_layer, shape, filter_shape, min_val, max_val, n_output):
    W_out = tf.Variable(tf.random_uniform(shape, min_val, max_val))
    b = tf.Variable(tf.zeros([n_output]))
    conv_2d = tf.nn.conv2d_transpose(
        value=prev_layer, filter=W_out,
        output_shape=tf.stack(filter_shape),
        strides=[1, 2, 2, 1], padding='SAME')
    return tf.add(conv_2d, b)


def fc_layer(prev_layer, in_dim, out_dim, output_shape):
    w_m = fc_weight_variable(in_dim, out_dim, output_shape)
    b_m = tf.Variable(tf.zeros([out_dim]))
    return tf.matmul(prev_layer, w_m) + b_m


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


def cross_entropy(y, target, offset=1e-7):
    obs_ = tf.clip_by_value(y, offset, 5 - offset)
    return -tf.reduce_sum(target * tf.log(obs_) + (1 - target) * tf.log(1 - obs_), [1, 2])


def l2_loss(y, target, offset=1e-7):
    obs_ = tf.clip_by_value(y, offset, 5 - offset)
    return tf.reduce_sum(tf.square(obs_ - target), [1, 2])


def l1_loss(y, target, offset=1e-7):
    obs_ = tf.clip_by_value(y, offset, 5 - offset)
    return tf.reduce_sum(tf.abs(obs_ - target), [1, 2])


def kl_div(z_mean, z_sigma, offset=1e-7):
    z_sigma_ = tf.clip_by_value(z_sigma, offset, 5 - offset)
    z_mean_ = tf.clip_by_value(z_mean, offset, 5 - offset)
    vae_loss_kl = -0.5 * tf.reduce_sum(1 + z_sigma_ - tf.square(z_mean_) - tf.exp(z_sigma_), 1)
    return vae_loss_kl
