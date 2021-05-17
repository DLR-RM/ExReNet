import time
import tensorflow as tf
from math import pi, tan
import numpy as np
from src.utils.TMatrix import TMatrix

class Utility:

    @staticmethod
    @tf.custom_gradient
    def norm_kdf(x):
        # Just computing the norm with a bit more stable gradients
        x = tf.sqrt(tf.reduce_sum(x * x, axis=1, keepdims=False) + 1e-19)

        def grad(dy):
            return dy * (x / (tf.norm(x, axis=1, keepdims=False) + 1.0e-19))

        return x, grad
