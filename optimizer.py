import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm, d_real, d_fake):
        preds_sub = preds
        labels_sub = labels

        self.real = d_real

        # Discrimminator Loss
        self.dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.real), logits=self.real, name='dclreal'))

        self.dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake, name='dcfake'))
        self.dc_loss = self.dc_loss_fake + self.dc_loss_real

        # Generator loss
        generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake, name='gl'))

        preds_sub = tf.reshape(preds_sub, [-1])

        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, labels=labels_sub, pos_weight=pos_weight))
        self.generator_loss = generator_loss + self.cost
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        all_variables = tf.compat.v1.trainable_variables()
        dc_var = [var for var in all_variables if 'dc_' in var.name]
        en_var = [var for var in all_variables if 'e_' in var.name]

        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()):
            self.discriminator_optimizer = tf.compat.v1.train.AdamOptimizer(
                learning_rate=FLAGS.discriminator_learning_rate,
                beta1=0.9, name='adam1').minimize(self.dc_loss,
                                                  var_list=dc_var)  # minimize(dc_loss_real, var_list=dc_var)

            self.generator_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
                                                                        beta1=0.9, name='adam2').minimize(
                self.generator_loss, var_list=en_var)

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

