import tensorflow as tf
from tensorflow.python.framework import function

dtype = tf.float32


@function.Defun(dtype, dtype, dtype, dtype)
def DoublySNGrad(logits, epsilon, dprev, dpout):
    prob = 1.0 / (1 + tf.exp(-logits))
    yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
    # {-1, 1} coding
    # yout = tf.sign(prob - epsilon)

    # unbiased
    dlogits = prob * (1 - prob) * (dprev + dpout)

    depsilon = dprev
    return dlogits, depsilon


@function.Defun(dtype, dtype, grad_func=DoublySNGrad)
def DoublySN(logits, epsilon):
    prob = 1.0 / (1 + tf.exp(-logits))
    yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
    return yout, prob


def compressive_binarize(logits):
    return