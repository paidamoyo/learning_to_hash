import os
import pprint
import sys

import numpy as np

from flags_parameters import set_params
from generate_data import generate
from  models.binary_ae_hashing import BinaryAEHashing
from  models.stochastic_generative_hashing import StochasticGenerativeHashing
from utils.metrics import euclidean_distance

if __name__ == '__main__':
    GPUID = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

    train_sgh = False
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
    data = generate(data=data_name)
    test_x, train_x = data['test_x'], data['train_x']
    euclidean = False
    if FLAGS.queries != test_x.shape[0] or euclidean:
        queries_idx = np.random.choice(np.arange(test_x.shape[0]), size=FLAGS.queries)
        test_queries = test_x[queries_idx]
        euclidean_distance(test_data=test_queries, train_data=train_x, data=data)
    else:
        test_queries = test_x
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
