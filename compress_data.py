import os
import pprint
import sys

import numpy as np
import scipy.io as sio

from flags_parameters import set_params
from  models.stochastic_generative_hashing import StochasticGenerativeHashing
from utils.metrics import euclidean_distance

if __name__ == '__main__':
    GPUID = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

    flags = set_params()
    FLAGS = flags.FLAGS
    np.random.seed(FLAGS.seed)
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__flags)

    args = sys.argv[1:]
    print("args:{}".format(args))
    if args:
        vm = float(args[0])
    else:
        vm = 1.0
    print("gpu_memory_fraction:{}".format(vm))

    train_data = sio.loadmat('dataset/mnist_training.mat')
    train_x = train_data['Xtraining']

    test_data = sio.loadmat('dataset/mnist_test.mat')
    test_x = test_data['Xtest']

    queries_idx = np.random.choice(np.arange(test_x.shape[0]), size=FLAGS.queries)
    test_queries = test_x[queries_idx]
    print("test_x:{}, train_x:{}, test_queries:{}".format(test_x.shape, train_x.shape, test_queries.shape))
    test_true_distance = euclidean_distance(test_data=test_queries, train_data=train_x)

    sgh = StochasticGenerativeHashing(batch_size=FLAGS.batch_size,
                                      learning_rate=FLAGS.learning_rate,
                                      beta1=FLAGS.beta1,
                                      beta2=FLAGS.beta2,
                                      num_iterations=FLAGS.num_iterations, seed=FLAGS.seed,
                                      l2_reg=FLAGS.l2_reg,
                                      input_dim=train_x.shape[1],
                                      num_examples=train_x.shape[0],
                                      latent_dim=FLAGS.latent_dim,
                                      train_x=train_x, test_x=test_x,
                                      alpha=FLAGS.alpha, test_queries=test_queries)

    with sgh.session:
        sgh.train_test()
