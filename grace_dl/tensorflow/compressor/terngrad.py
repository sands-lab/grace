import tensorflow as tf

from grace_dl.tensorflow import Compressor


class TernGradCompressor(Compressor):
    """
    (2017). TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning.
    Retrieved from http://arxiv.org/abs/1705.07878
    """
    def compress(self, tensor, name):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])

        std = tf.math.square(tensor_flatten - tf.math.reduce_mean(tensor_flatten))
        std = tf.math.sqrt(tf.math.reduce_mean(std))
        c = 2.5
        gradient = tf.clip_by_value(tensor_flatten, -c * std, c * std)
        scaler = tf.math.reduce_max(tf.math.abs(gradient))

        zeros = tf.zeros(tf.shape(tensor_flatten))
        abs_gradient = tf.abs(gradient)
        sign_gradient = tf.sign(gradient)
        rnd_sample = tf.random.uniform(tf.shape(tensor_flatten), 0, scaler)
        where_cond = tf.less(rnd_sample, abs_gradient)
        binarized_gradient = tf.where(where_cond, sign_gradient * scaler, zeros)
        new_sign = tf.sign(binarized_gradient)  # -1, 0, 1

        scaler = tf.reshape(scaler, [-1])
        ctx = tensor_shape
        tensor_compressed = tf.cast(new_sign, tf.int8), scaler

        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx):
        tensor_shape = ctx
        tensor_compressed, scaler = tensor_compressed

        sign = tf.cast(tensor_compressed, dtype=tf.float32)
        tensor_decompressed = sign * scaler
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed
