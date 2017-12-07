import logging

import tensorflow as tf

from models.stochastic_generative_hashing import StochasticGenerativeHashing


class BinaryAEHashing(StochasticGenerativeHashing):
    def __init__(self, alpha, l2_reg, batch_size, latent_dim, seed, num_iterations, learning_rate, beta1, beta2,
                 train_x, test_x, test_queries, input_dim, num_examples):
        StochasticGenerativeHashing.__init__(self, alpha, l2_reg, batch_size, latent_dim, seed, num_iterations,
                                             learning_rate, beta1, beta2, train_x, test_x, test_queries, input_dim,
                                             num_examples)

        self.log_file = 'binary_hashing.log'
        self.model_results = 'BAEH_mnsit_'
        self.is_stochastic = True

    def _objective(self):
        self._build_model()
        self.num_batches = self.num_examples / self.batch_size
        logging.debug("num batches:{}, batch_size:{} epochs:{}".format(self.num_batches, self.batch_size,
                                                                       int(self.num_iterations / self.num_batches)))
        self.x_recon_loss = tf.losses.mean_squared_error(predictions=self.x_recon, labels=self.x)
        z_recon_loss = tf.losses.mean_squared_error(predictions=self.p_out, labels=self.y_out)
        w_decode_reg = self.l2_reg * tf.nn.l2_loss(self.w_decode) / self.batch_size
        w_encode_reg = self.l2_reg * tf.nn.l2_loss(self.w_encode) / self.batch_size
        self.cost = self.x_recon_loss + self.alpha * z_recon_loss + w_decode_reg + w_encode_reg

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = optimizer.minimize(self.cost)
