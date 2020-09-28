import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('discriminator_out', 0, 'discriminator_out.')
flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', .5 * 0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 1, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('output_dimension', 35, 'Second dimension of output.')
flags.DEFINE_integer('multi_view_K', 13, 'K param for multi-view')
flags.DEFINE_integer('single_view_K', 7, 'K param for single-view')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seed', 50, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 30, 'number of iterations.')

seed = 7


def get_settings_new(model, view):
    iterations = FLAGS.iterations
    if view == -1:
        FLAGS.output_dimension = 140
        FLAGS.hidden1 = 70
    re = {'iterations': iterations, 'model': model}

    return re
