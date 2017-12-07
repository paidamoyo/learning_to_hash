import numpy as np
import tensorflow as tf
from tensorflow.python.framework import function

dtype = tf.float32


@function.Defun(dtype, dtype, dtype, dtype)
def DoublySNGrad(logits, epsilon, dprev, dpout):
    prob = tf.sigmoid(logits)
    yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
    # {-1, 1} coding
    # yout = tf.sign(prob - epsilon)
    # unbiased
    dlogits = prob * (1 - prob) * (dprev + dpout)

    depsilon = dprev
    return dlogits, depsilon


@function.Defun(dtype, dtype, grad_func=DoublySNGrad)
def DoublySN(logits, epsilon):
    prob = tf.sigmoid(logits)
    yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
    return yout, prob


def stochastic_neuron(logits, latent_dim, batch_size_tensor):
    prob = tf.sigmoid(logits)
    ones = np.ones(shape=latent_dim, dtype=np.float32)
    epsilon = tf.distributions.Uniform(low=ones * 0, high=ones).sample(sample_shape=[batch_size_tensor])
    y_out = (tf.sign(prob - epsilon) + 1.0) / 2
    return y_out, prob
