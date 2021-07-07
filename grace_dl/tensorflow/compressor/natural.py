import tensorflow as tf

from grace_dl.tensorflow import Compressor


class NaturalCompressor(Compressor):
    """
    (2019). Natural compression for distributed deep learning.
    Retrieved from https://arxiv.org/abs/1905.10988
    """
    def compress(self, tensor, name):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])

        tensor_cast = tf.bitcast(tensor_flatten, tf.int32)
        sign = tf.bitwise.bitwise_and(tensor_cast, 0b10000000000000000000000000000000)
        exp = tf.bitwise.bitwise_and(tensor_cast, 0b01111111100000000000000000000000)
        mantissa = tf.bitwise.bitwise_and(tensor_cast, 0b00000000011111111111111111111111)
        exp_add_one = mantissa > tf.random.uniform(tf.shape(tensor_flatten), minval=0, maxval=0x007fffff,
                                                   dtype=tf.int32)
        # exp_add_one = mantissa > 0x00400000 # deterministic
        exponent = tf.where(exp_add_one, exp + 0b00000000100000000000000000000000, exp)
        # original exponent range: -128 ~ 127, clip to -110,  17
        # view as uint8_t:            0 ~ 255            18  145
        exp_shift = tf.clip_by_value(exponent, 0b00001001000000000000000000000000, 0b01001000100000000000000000000000)
        exps = tf.bitwise.right_shift(exp_shift, 23)
        # shift 18 so that 0 maps to -110 and 127 maps to 145
        # set MSB if negative
        exps = tf.bitwise.bitwise_or(tf.bitwise.right_shift(sign, 24), exps - 18)
        tensor_compressed = tf.cast(exps, tf.uint8)

        return [tensor_compressed], tensor_shape

    def decompress(self, tensors_compressed, tensor_shape):
        tensor_compressed, = tensors_compressed
        sign = tensor_compressed > 127
        exps = tf.bitwise.bitwise_and(tensor_compressed, 0b01111111)
        exps_shift = tf.bitwise.left_shift(tf.cast(exps + 18, tf.int32), 23)
        floats = tf.bitcast(exps_shift, tf.float32)
        tensor_decompressed = tf.where(sign, -floats, floats)
        tensor_decompressed = tf.multiply(tf.cast(exps >= 1, tensor_decompressed.dtype), tensor_decompressed)
        return tf.reshape(tensor_decompressed, tensor_shape)
