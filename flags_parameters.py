import tensorflow as tf


def set_params():
    flags = tf.app.flags
    flags.DEFINE_integer("num_iterations", 40000, "DASA number of iterations")
    flags.DEFINE_integer("batch_size", 350, "Batch size")
    flags.DEFINE_integer("dropout_rate", 0.0, "denoising input dropout rate")
    flags.DEFINE_integer("seed", 31415, "random seed")
    flags.DEFINE_integer("require_improvement", 10000, "num of iterations before early stopping")
    flags.DEFINE_integer("learning_rate", 3e-4, "optimizer learning rate")
    flags.DEFINE_float("beta1", 0.9, "optimizer beta 1")
    flags.DEFINE_float("beta2", 0.999, "optimizer beta 2")
    flags.DEFINE_integer("hidden_dim", [50, 50], "hidden layer dimensions and size")
    flags.DEFINE_string("risk_function", 'NA', "risk function is not simulated [linear, gaussian, NA]")
    flags.DEFINE_integer("latent_dim", 50, "latent dimensions of z")
    flags.DEFINE_float("l2_reg", 0.001, "l2 regularization weight multiplier (just for debugging not optimization)")
    flags.DEFINE_float("l1_reg", 0.001, "l1 regularization weight multiplier (just for debugging not optimization)")
    flags.DEFINE_float("keep_prob", 0.9, "keep prob for weights implementation in layers")
    flags.DEFINE_integer("sample_size", 157, "number of samples of generated time")
    flags.DEFINE_float("lambda_trecon", 10, "weight of time reconstruction")
    # TODO split keep prob of denoising and risk
    # TODO split generator updates?
    # TODO test different denoising dropouts
    flags.DEFINE_integer("disc_updates", 1, "number of discriminator updates before generator update")
    return flags
