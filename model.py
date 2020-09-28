from layers import HypergraphConvolution
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.compat.v1.variable_scope(self.name):
            self._build()
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class HCAE(Model):
    def __init__(self, placeholders, H_dim, **kwargs):
        super(HCAE, self).__init__(**kwargs)

        self.input_dim = H_dim
        self.H = placeholders['H']
        self.dropout = placeholders['dropout']
        self.G = placeholders['G']
        self.build()

    def _build(self):
        with tf.compat.v1.variable_scope('Encoder', reuse=tf.compat.v1.AUTO_REUSE):
            self.hidden1 = tf.nn.relu(HypergraphConvolution(input_dim=self.input_dim,
                                                            output_dim=FLAGS.hidden1,
                                                            G=self.G,
                                                            dropout=self.dropout,
                                                            act=lambda x: x,
                                                            logging=self.logging,
                                                            name='e_dense_1')(self.H))

            self.noise = gaussian_noise_layer(self.hidden1, 0.1)

            self.embeddings = HypergraphConvolution(input_dim=FLAGS.hidden1,
                                                    output_dim=FLAGS.hidden2,
                                                    G=self.G,
                                                    dropout=self.dropout,
                                                    act=lambda x: x,
                                                    logging=self.logging,
                                                    name='e_dense_2')(self.hidden1)

            self.z_mean = self.embeddings

            self.reconstructions = dense(self.embeddings, 1, FLAGS.output_dimension, name='e_dense_3')


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.compat.v1.variable_scope(name, reuse=None):
        tf.compat.v1.set_random_seed(1)
        weights = tf.compat.v1.get_variable("weights", shape=[n1, n2],
                                            initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.compat.v1.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


class Discriminator(Model):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__(**kwargs)

        self.act = tf.nn.relu

    def construct(self, inputs, reuse=False):
        with tf.compat.v1.variable_scope('Discriminator'):
            if reuse:
                tf.compat.v1.get_variable_scope().reuse_variables()
            tf.compat.v1.set_random_seed(1)
            dc_den1 = tf.nn.relu(dense(inputs, FLAGS.hidden2, FLAGS.hidden3, name='dc_den1'))
            dc_den2 = tf.nn.relu(dense(dc_den1, FLAGS.hidden3, FLAGS.hidden1, name='dc_den2'))
            output = dense(dc_den2, FLAGS.hidden1, 1, name='dc_output')
            return output


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise
