import logging

import tensorflow as tf

from models.stochastic_generative_hashing import StochasticGenerativeHashing
from utils.quantization import ba_binarize


class BinaryAEHashing(StochasticGenerativeHashing):
    def __init__(self, alpha, l2_reg, batch_size, latent_dim, seed, num_iterations, learning_rate, beta1, beta2,
                 train_x, test_x, test_queries, input_dim, num_examples):
        StochasticGenerativeHashing.__init__(self, alpha, l2_reg, batch_size, latent_dim, seed, num_iterations,
                                             learning_rate, beta1, beta2, train_x, test_x, test_queries, input_dim,
                                             num_examples)

        self.log_file = 'binary_hashing.log'
        self.model_results = 'BAEH_mnsit_'

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

    def _decoder(self):
        with tf.name_scope('decode'):
            self.w_decode = tf.Variable(
                tf.random_normal([self.latent_dim, self.input_dim], stddev=1.0 / tf.sqrt(float(self.latent_dim)),
                                 dtype=self.dtype), name='w_decode')

        with tf.name_scope('scale'):
            scale_para = tf.Variable(tf.constant(self.train_var, dtype=self.dtype), name="scale_para")
            shift_para = tf.Variable(tf.constant(self.train_mean, dtype=self.dtype), name="shift_para")

        self.x_recon = tf.matmul(self.y_out, self.w_decode) * tf.abs(scale_para) + shift_para

    def _encoder(self):
        with tf.name_scope('encode'):
            self.w_encode = tf.Variable(
                tf.random_normal([self.input_dim, self.latent_dim], stddev=1.0 / tf.sqrt(float(self.input_dim)),
                                 dtype=self.dtype), name='w_encode')
            b_encode = tf.Variable(tf.random_normal([self.latent_dim], dtype=self.dtype), name='b_encode')
            self.h_encode = tf.matmul(self.x, self.w_encode) + b_encode
            # determinastic output
        h_epsilon = tf.ones(shape=tf.shape(self.h_encode), dtype=self.dtype) * .5
        self.y_out, self.p_out = ba_binarize(logits=self.h_encode, epsilon=h_epsilon)
