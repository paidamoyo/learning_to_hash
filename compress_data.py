import os
import pprint
import sys

import numpy as np

from flags_parameters import set_params
from  models.binary_ae_hashing import BinaryAEHashing
from  models.stochastic_generative_hashing import StochasticGenerativeHashing
from utils.generate_data import generate
from utils.metrics import euclidean_distance

if __name__ == '__main__':
    GPUID = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

    train_sgh = True
    if train_sgh:
        model = StochasticGenerativeHashing
    else:
        model = BinaryAEHashing

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

    data_name = 'cifar'
    all_data = generate(data=data_name)
    test_x, train_x = all_data['test_x'], all_data['train_x']
    compute_nn = False
    if compute_nn:
        queries_idx = np.random.choice(np.arange(test_x.shape[0]), size=FLAGS.queries)
        test_queries = test_x[queries_idx]
        np.save('results/test_queries_{}'.format(data_name), test_queries)
        euclidean_distance(test_data=test_queries, train_data=train_x, data=data_name)
    elif data_name == 'mnsit':
        test_queries = test_x
    elif data_name == 'cifar':
        test_queries = np.load("results/test_queries_{}.npy".format(data_name))
    print("test_x:{}, train_x:{}, test_queries:{}".format(test_x.shape, train_x.shape, test_queries.shape))

    hash = model(batch_size=FLAGS.batch_size,
                 learning_rate=FLAGS.learning_rate,
                 beta1=FLAGS.beta1,
                 beta2=FLAGS.beta2,
                 num_iterations=FLAGS.num_iterations, seed=FLAGS.seed,
                 l2_reg=FLAGS.l2_reg,
                 input_dim=train_x.shape[1],
                 num_examples=train_x.shape[0],
                 latent_dim=FLAGS.latent_dim,
                 train_x=train_x, test_x=test_x,
                 alpha=FLAGS.alpha, test_queries=test_queries, data=data_name)

    with hash.session:
        hash.train_test()
