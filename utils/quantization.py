import tensorflow as tf
from tensorflow.python.framework import function

dtype = tf.float32


@function.Defun(dtype, dtype, dtype, dtype)
def doubly_SN_grad(logits, epsilon, dprev, dpout):
    prob = tf.sigmoid(logits)
    yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
    # {-1, 1} coding
    # yout = tf.sign(prob - epsilon)
    # unbiased
    dlogits = prob * (1 - prob) * (dprev + dpout)

    depsilon = dprev
    return dlogits, depsilon


@function.Defun(dtype, dtype, grad_func=doubly_SN_grad)
def doubly_SN(logits, epsilon):
    prob = tf.sigmoid(logits)
    yout = (tf.sign(prob - epsilon) + 1.0) / 2.0
    return yout, prob
