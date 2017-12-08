import logging
import os
import time
from datetime import timedelta

import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils.generate_data import reshape_mnsit, reshape_cifar
from utils.metrics import plot_cost, recall_n, plot_recon
from utils.quantization import doubly_SN


class StochasticGenerativeHashing(object):
    def __init__(self, alpha, l2_reg, batch_size, latent_dim, seed, num_iterations,
                 learning_rate, beta1, beta2, train_x, test_x, test_queries, input_dim, num_examples, data):
        self.l2_reg = l2_reg
        self.alpha = alpha
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.seed = seed
        self.num_iterations = num_iterations
        self.learning_rate, self.beta1, self.beta2 = learning_rate, beta1, beta2
        self.log_file = 'stochastic_generative_hashing.log'
        self.data = data
        self.model_results = 'SGH_{}_'.format(self.data)
        logging.basicConfig(filename=self.log_file, filemode='w', level=logging.DEBUG)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.batch_norm = True
        self.is_stochastic = True

        self.config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True
        # self.config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        # Load Data
        self.train_x = train_x
        self.train_mean = self.train_x.mean(axis=0).astype('float64')
        self.train_var = np.clip(self.train_x.var(axis=0), 1e-7, np.inf).astype('float64')
        train_statistics = "train_mean:{}, train_var:{}".format(self.train_mean, self.train_var)
        # print(train_statistics)
        logging.debug(train_statistics)

        self.test_x = test_x
        self.test_queries = test_queries
        self.input_dim = input_dim
        # self.imputation_values = imputation_values
        self.imputation_values = np.zeros(shape=self.input_dim)
        self.num_examples = num_examples
        self.dtype = tf.float32

        self._build_graph()
        self.train_cost, self.train_recon = [], []

    def _build_graph(self):
        self.G = tf.Graph()
        with self.G.as_default():
            self.x = tf.placeholder(self.dtype, shape=[None, self.input_dim], name='x')
            self.batch_size_tensor = tf.placeholder(tf.int32, shape=[], name='batch_size')
            self.stochastic = tf.placeholder(tf.bool)
            self._objective()
            self.session = tf.Session(config=self.config)

            self.saver = tf.train.Saver()
            self.merged = tf.summary.merge_all()
            self.current_dir = os.getcwd()
            self.save_path = self.current_dir + "/summaries/" + self.model_results
            self.train_writer = tf.summary.FileWriter(self.save_path, self.session.graph)

    def _objective(self):
        self._build_model()
        self.num_batches = self.num_examples / self.batch_size
        logging.debug("num batches:{}, batch_size:{} epochs:{}".format(self.num_batches, self.batch_size,
                                                                       int(self.num_iterations / self.num_batches)))
        self.x_recon_loss = tf.losses.mean_squared_error(predictions=self.x_recon, labels=self.x)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.p_out, labels=self.y_out))
        w_decode_reg = self.l2_reg * tf.nn.l2_loss(self.w_decode) / self.batch_size
        w_encode_reg = self.l2_reg * tf.nn.l2_loss(self.w_encode) / self.batch_size
        self.cost = self.x_recon_loss + self.alpha * cross_entropy_loss + w_decode_reg + w_encode_reg

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = optimizer.minimize(self.cost)

    def _build_model(self):
        self._encoder()
        self._decoder()

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
        self.y_out, self.p_out = doubly_SN(self.h_encode, self.h_epsilon())

    def train_neural_network(self):
        train_print = "Training {} Model:".format(self.model_results)
        params_print = "Parameters:, l2_reg:{}, learning_rate:{}," \
                       " momentum: beta1={} beta2={}, batch_size:{}, batch_norm:{}," \
                       "latent_dim:{}, num_of_batches:{}, stochastic:{}, data:{}" \
            .format(self.l2_reg, self.learning_rate, self.beta1, self.beta2, self.batch_size,
                    self.batch_norm, self.latent_dim, self.num_batches, self.is_stochastic, self.data)
        print(train_print)
        print(params_print)
        logging.debug(train_print)
        logging.debug(params_print)
        self.session.run(tf.global_variables_initializer())
        learning_rate = self.learning_rate

        start_time = time.time()
        self.show_all_variables()

        for i in range(self.num_iterations):
            # Batch Training
            indx = np.random.choice(self.input_dim, self.batch_size)
            x_batch = self.train_x[indx]
            # Ending time.
            _, x_recon_loss, batch_cost = self.session.run(
                [self.optimize, self.x_recon_loss, self.cost],
                feed_dict={self.x: x_batch,
                           self.batch_size_tensor: self.batch_size,
                           self.stochastic: self.is_stochastic})

            if i % 100 == 0:
                print_iteration = 'Num iteration: %d Total Loss: %0.04f Recon Loss %0.04f' % (
                    i, batch_cost, x_recon_loss)
                print(print_iteration)
                logging.debug(print_iteration)
                self.train_cost.append(batch_cost)
                self.train_recon.append(x_recon_loss)

            if i % 2000 == 0:
                learning_rate = 0.5 * learning_rate

        self.saver.save(sess=self.session, save_path=self.save_path)
        t_vars = tf.trainable_variables()

        para_list = {}
        for var in t_vars:
            para_list[var.name] = self.session.run(var)

        end_time = time.time()
        time_dif = end_time - start_time
        time_dif_print = "Time usage: " + str(timedelta(seconds=int(round(time_dif))))
        print(time_dif_print)
        logging.debug(time_dif_print)
        # shutdown everything to avoid zombies
        return end_time, para_list

    def train_test(self):
        end_time, para_list = self.train_neural_network()
        W = para_list['encode/w_encode:0']
        b = para_list['encode/b_encode:0']
        U = para_list['decode/w_decode:0']
        shift = para_list['scale/shift_para:0']
        scale = para_list['scale/scale_para:0']
        self.saver.restore(sess=self.session, save_path=self.save_path)

        test_xhat, test_recon_loss, test_cost, y_test = self.session.run(
            [self.x_recon, self.x_recon_loss, self.cost, self.y_out],
            feed_dict={self.x: self.test_x,
                       self.batch_size_tensor: self.test_x.shape[
                           0],
                       self.stochastic: self.is_stochastic})

        y_test_queries = self.session.run(
            self.y_out, feed_dict={self.x: self.test_queries,
                                   self.batch_size_tensor: self.test_queries.shape[0],
                                   self.stochastic: self.is_stochastic})

        size = 30
        if self.data == 'mnsit':
            template = np.hstack([np.vstack([reshape_mnsit(j, self.test_x), reshape_mnsit(j, test_xhat)
                                             ]) for j in range(size)])
        else:
            template = np.hstack(
                [np.vstack([reshape_cifar(j, self.test_x), reshape_cifar(j, test_xhat)
                            ]) for j in range(size)])

        train_xhat, train_recon_loss, train_cost, y_train = self.session.run(
            [self.x_recon, self.x_recon_loss, self.cost, self.y_out],
            feed_dict={self.x: self.train_x,
                       self.batch_size_tensor:
                           self.train_x.shape[0],
                       self.stochastic: self.is_stochastic})

        print_cost = "Train: recon:{}, cost:{}, Test: recon:{}, cost:{}".format(train_recon_loss, train_cost,
                                                                                test_recon_loss, test_cost)
        logging.debug(print_cost)
        print(print_cost)

        filename = 'results/' + self.model_results + str(self.latent_dim) + 'bit.mat'

        plot_cost(self.train_cost)
        plot_recon(template=template)
        test_recall = recall_n(test_data=y_test_queries, train_data=y_train, data=self.data)
        sio.savemat(filename,
                    {'y_train': y_train, 'y_test': y_test, 'train_time': end_time,
                     'W_encode': W, 'b_encode': b, 'U': U,
                     'shift': shift, 'scale': scale,
                     'train_cost': self.train_cost, 'test_recon': test_recon_loss,
                     'test_cost': test_cost,
                     'test_recall': test_recall, 'test_x': self.test_x[0: size],
                     'test_xhat': test_xhat[0: size]})  # define doubly stochastic neuron with gradient by DeFun

    @staticmethod
    def show_all_variables():
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    def h_epsilon(self):
        epsilon_det = tf.ones(shape=tf.shape(self.h_encode), dtype=self.dtype) * .5
        one = np.ones(shape=self.latent_dim, dtype=np.float32)
        print("z_ones:{}".format(one.shape))
        epsilon_stoc = tf.distributions.Uniform(low=one * 0, high=one).sample(
            sample_shape=[self.batch_size_tensor])
        return tf.cond(self.stochastic, lambda: epsilon_stoc, lambda: epsilon_det)
