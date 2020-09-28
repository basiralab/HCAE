from __future__ import division
from __future__ import print_function
import tensorflow as tf
from constructor import format_hyper_data, get_hyper_placeholder, get_hyper_model, get_hyper_optimizer, hyper_update
import os

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS


class Encoder():
    def __init__(self, settings):
        self.iteration = settings['iterations']
        self.model = settings['model']
        self.cost = 0

    def erun(self, H, G):
        tf.compat.v1.reset_default_graph()
        model_str = self.model
        print(model_str)
        # formatted data
        feas = format_hyper_data(H, G)

        # Define placeholders
        placeholders = get_hyper_placeholder(feas['H'])

        # construct model
        d_real, discriminator, ae_model = get_hyper_model(model_str, placeholders, int(feas['H_dim'][1]))

        # Optimizer
        opt = get_hyper_optimizer(model_str, ae_model, discriminator, placeholders, feas['pos_weight'], feas['norm'], H,
                                  d_real)

        # Initialize session
        sess = tf.compat.v1.Session()
        sess.run(tf.global_variables_initializer())

        # Train model
        for epoch in range(self.iteration):  # self.iteration

            emb, avg_cost = hyper_update(ae_model, opt, sess, feas['H'], feas['H_orig'], feas['G'], placeholders, G)
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))

        self.cost += avg_cost

        return emb
