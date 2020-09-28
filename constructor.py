import tensorflow as tf
import numpy as np
from model import Discriminator, HCAE
from optimizer import OptimizerAE
import inspect
from scipy.stats import norm

from preprocessing import construct_feed_dict, construct_hyper_feed_dict

flags = tf.app.flags
FLAGS = flags.FLAGS


def get_hyper_placeholder(H):
    placeholders = {
        'H': tf.compat.v1.placeholder(tf.float32),
        'G': tf.compat.v1.placeholder(tf.float32),
        'H_orig': tf.compat.v1.placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'real_distribution': tf.compat.v1.placeholder(dtype=tf.float32, shape=[H.shape[1], FLAGS.hidden2],
                                                      name='real_distribution')

    }

    return placeholders


def get_hyper_model(model_str, placeholders, H_dim):
    discriminator = Discriminator()
    d_real = discriminator.construct(placeholders['real_distribution'])
    model = None
    if model_str == 'hyper_arga_ae':
        model = HCAE(placeholders, H_dim)

    return d_real, discriminator, model


def format_hyper_data(H, G):
    H_orig = H

    pos_weight = float(H.shape[0] * H.shape[1] - H.sum()) / H.sum()
    norm = H.shape[0] * H.shape[1] / float((H.shape[0] * H.shape[1] - H.sum()) * 2)
    H_dim = [H.shape[0], H.shape[1]]

    values = [H, G, H_dim, H_orig, pos_weight, norm]

    keys = ['H', 'G', 'H_dim', 'H_orig', 'pos_weight', 'norm']
    feas = dict(zip(keys, values))

    return feas


def get_hyper_optimizer(model_str, model, discriminator, placeholders, pos_weight, norm, H_orig, d_real):
    if model_str == 'hyper_arga_ae':
        d_fake = discriminator.construct(model.embeddings, reuse=True)
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(placeholders['H_orig'], [-1]),
                          pos_weight=pos_weight,
                          norm=norm,
                          d_real=d_real,
                          d_fake=d_fake)
    return opt


def hyper_update(model, opt, sess, H, H_orig, G, placeholders, prior):
    # Construct feed dictionary
    feed_dict = construct_hyper_feed_dict(H, H_orig, G, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    feed_dict.update({placeholders['dropout']: 0})
    emb = sess.run(model.z_mean, feed_dict=feed_dict)
    featureAverage = np.mean(prior, axis=1)
    (mu, sigma) = norm.fit(featureAverage)

    z_real_dist_prior = np.random.normal(mu, sigma, (H.shape[1], FLAGS.hidden2))

    feed_dict.update({placeholders['real_distribution']: z_real_dist_prior})

    for j in range(5):
        _, reconstruct_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
    d_loss, _ = sess.run([opt.dc_loss, opt.discriminator_optimizer], feed_dict=feed_dict)
    g_loss, _ = sess.run([opt.generator_loss, opt.generator_optimizer], feed_dict=feed_dict)

    avg_cost = reconstruct_loss

    return emb, avg_cost


def predict(model, opt, sess, adj_norm, adj_label, features, placeholders, adj, prior):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    feed_dict.update({placeholders['dropout']: 0})
    emb = sess.run(model.z_mean, feed_dict=feed_dict)

    return emb


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]
