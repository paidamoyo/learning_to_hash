import tensorflow as tf


def set_params():
    flags = tf.app.flags
    flags.DEFINE_integer("num_iterations", 1, "number of iterations")
    flags.DEFINE_integer("batch_size", 500, "Batch size")
    flags.DEFINE_integer("seed", 31415, "random seed")
    flags.DEFINE_integer("learning_rate", 1e-2, "optimizer learning rate")
    flags.DEFINE_float("beta1", 0.9, "optimizer beta 1")
    flags.DEFINE_float("beta2", 0.999, "optimizer beta 2")
    flags.DEFINE_integer("latent_dim", 64, "latent dimensions of z")
    flags.DEFINE_float("l2_reg", 1e-3, "l2 regularization weight multiplier (just for debugging not optimization)")
    flags.DEFINE_float('alpha', 1e-3, "cross entropy loss weight")
    flags.DEFINE_integer('queries', 100, "size of test queries")

