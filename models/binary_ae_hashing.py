import logging
import os
import time
from datetime import timedelta

import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils.metrics import plot_cost, recall_n
from utils.quantization import DoublySN


class BinaryAEHashing(object):
    def __init__(self, alpha, l2_reg, batch_size, latent_dim, seed, num_iterations,
                 learning_rate, beta1, beta2, train_x, test_x, test_queries, input_dim, num_examples):
        self.l2_reg = l2_reg
        self.alpha = alpha
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.seed = seed
        self.num_iterations = num_iterations
        self.learning_rate, self.beta1, self.beta2 = learning_rate, beta1, beta2
        self.log_file = 'stochastic_generative_hashing.log'
        logging.basicConfig(filename=self.log_file, filemode='w', level=logging.DEBUG)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        self.batch_norm = True

        self.config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        self.config.gpu_options.allow_growth = True
        # self.config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
        # Load Data
        self.train_x = train_x
        self.train_mean = self.train_x.mean(axis=0).astype('float64')
        self.train_var = np.clip(self.train_x.var(axis=0), 1e-7, np.inf).astype('float64')
        train_statistics = "train_mean:{}, train_var:{}".format(self.train_mean, self.train_var)
        print(train_statistics)
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
            self._objective()
            self.session = tf.Session(config=self.config)

            self.saver = tf.train.Saver()
            self.merged = tf.summary.merge_all()
            self.current_dir = os.getcwd()
            self.save_path = self.current_dir + "/summaries/sgh_model"
            self.train_writer = tf.summary.FileWriter(self.save_path, self.session.graph)

    def _objective(self):
        self._build_model()
        self.num_batches = self.num_examples / self.batch_size
        logging.debug("num batches:{}, batch_size:{} epochs:{}".format(self.num_batches, self.batch_size,
                                                                       int(self.num_iterations / self.num_batches)))
        self.x_recon_loss = tf.nn.l2_loss(self.x_recon - self.x, name=None)
        cross_entropy_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.h_encode, labels=self.y_out))
        w_decode_reg = self.l2_reg * tf.nn.l2_loss(self.w_decode, name=None)
        w_encode_reg = self.l2_reg * tf.nn.l2_loss(self.w_encode, name=None)
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
            # determinastic output
            h_epsilon = tf.ones(shape=tf.shape(self.h_encode), dtype=self.dtype) * .5
        self.y_out, self.pout = DoublySN(self.h_encode, h_epsilon)

    def train_neural_network(self):
        train_print = "Training DASA Model:"
        params_print = "Parameters:, l2_reg:{}, learning_rate:{}," \
                       " momentum: beta1={} beta2={}, batch_size:{}, batch_norm:{}," \
                       "latent_dim:{}, num_of_batches:{}" \
            .format(self.l2_reg, self.learning_rate, self.beta1, self.beta2, self.batch_size,
                    self.batch_norm, self.latent_dim, self.num_batches)
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
            batch_size = len(x_batch)
            # Ending time.
            _, x_recon_loss, batch_cost = self.session.run([self.optimize, self.x_recon_loss, self.cost],
                                                           feed_dict={self.x: x_batch})

            if i % 100 == 0:
                print_iteration = 'Num iteration: %d Total Loss: %0.04f Recon Loss %0.04f' % (
                    i, batch_cost / batch_size, x_recon_loss / batch_size)
                print(print_iteration)
                logging.debug(print_iteration)
                self.train_cost.append(batch_cost)
                self.train_recon.append(x_recon_loss)

            if i % 2000 == 0:
                learning_rate = 0.5 * learning_rate

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
        epsilon = 0.5

        # Test Querries
        test_logits = np.dot(np.array(self.test_queries), W) + b
        pres = 1.0 / (1 + np.exp(-test_logits))
        h_test = (np.sign(pres - epsilon) + 1.0) / 2.0

        # Train
        train_logits = np.dot(np.array(self.train_x), W) + b
        train_pres = 1.0 / (1 + np.exp(-train_logits))
        h_train = (np.sign(train_pres - epsilon) + 1.0) / 2.0

        filename = 'results/SGH_mnist_' + str(self.latent_dim) + 'bit.mat'
        sio.savemat(filename,
                    {'h_train': h_train, 'h_test': h_test, 'train_time': end_time,
                     'W_encode': W, 'b_encode': b, 'U': U,
                     'shift': shift, 'scale': scale,
                     'train_cost': self.train_cost})  # define doubly stochastic neuron with gradient by DeFun

        plot_cost(self.train_cost)
        recall_n(test_data=h_test, train_data=h_train)

    @staticmethod
    def show_all_variables():
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)
