import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        rank = len(z_mean.shape)
        
        if rank == 2:  # 2D case
            dim = tf.shape(z_mean)[1]
            epsilon_shape = (batch, dim)
        elif rank == 1:  # 1D case
            epsilon_shape = (batch,)
        elif rank == 3:  # 3D case
            dim1 = tf.shape(z_mean)[1]
            dim2 = tf.shape(z_mean)[2]
            epsilon_shape = (batch, dim1, dim2)
        else:
            raise ValueError("z_mean and z_log_var must be 1D, 2D, or 3D tensors")
        
        epsilon = tf.keras.backend.random_normal(shape=epsilon_shape)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def kl_divergence_sum(mu1 = 0.0, log_var1 = 0.0, mu2 = 0.0, log_var2 = 0.0):
    var1 = tf.exp(log_var1)
    var2 = tf.exp(log_var2)
    axis0 = 0.5*tf.reduce_mean(log_var2 - log_var1 + (var1 + (mu1 - mu2)**2) / var2 - 1, axis=0)
    return tf.reduce_sum(axis0)



def log_lik_normal_sum(x, mu=0.0, log_var = 0.0):
    axis0 = -0.5*(tf.math.log(2*np.pi) + tf.reduce_mean(log_var + (x - mu) ** 2 * tf.exp(-log_var), axis=0))
    return tf.reduce_sum(axis0)

