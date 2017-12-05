import os
import pprint
import sys

import scipy.io as sio

from flags_parameters import set_params
from  models.stochastic_generative_hashing import StochasticGenerativeHashing

if __name__ == '__main__':
    GPUID = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)

    flags = set_params()
    FLAGS = flags.FLAGS
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
    test_data = sio.loadmat('dataset/mnist_test.mat')
    train_x = train_data['Xtraining']
    test_x = test_data['Xtest']
    print("test_x:{}, train__x:{}".format(test_x.shape, train_x.shape))

    sgh = StochasticGenerativeHashing(batch_size=FLAGS.batch_size,
                                      learning_rate=FLAGS.learning_rate,
                                      beta1=FLAGS.beta1,
                                      beta2=FLAGS.beta2,
                                      num_iterations=FLAGS.num_iterations, seed=FLAGS.seed,
                                      l2_reg=FLAGS.l2_reg,
                                      input_dim=train_x.shape[1],
                                      num_examples=train_x.shape[0],
                                      latent_dim=FLAGS.latent_dim, train_x=train_x, test_x=test_x, alpha=FLAGS.alpha)

    with sgh.session:
        sgh.train_test()
