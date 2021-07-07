import math

import tensorflow as tf

from grace_dl.tensorflow import Compressor


class INCEPTIONNCompressor(Compressor):
    """
    (2018). A network-centric hardware/algorithm co-design to accelerate distributed training of deep neural networks.
    https://doi.org/10.1109/MICRO.2018.00023
    """
    def __init__(self, error_bound):
        super().__init__(tensors_size_are_same=False)
        self.error_bound = error_bound

    def compress(self, tensor, name):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        tensor_cast = tf.bitcast(tensor_flatten, tf.uint32)
        sign = tf.bitwise.bitwise_and(tensor_cast, 0b10000000000000000000000000000000)
        exp = tf.bitwise.bitwise_and(tensor_cast, 0b01111111100000000000000000000000)
        mantissa = tf.bitwise.bitwise_and(tensor_cast, 0b00000000011111111111111111111111)

        exp = tf.bitwise.right_shift(exp, 23)

        error_bound_val = self.error_bound
        error_bound = 127 + int(math.log(error_bound_val / 2, 10))  # error_bound exponent: 117 for 2e-10
        radius = math.ceil((127 - error_bound) / 2)
        mid = error_bound + radius
        mask_32bit = exp >= 127
        mask_16bit = (exp >= mid) & (exp < 127)
        mask_8bit = (exp >= error_bound) & (exp < mid)
        indices_32bit = tf.reshape(tf.where(mask_32bit), [-1])
        indices_16bit = tf.reshape(tf.where(mask_16bit), [-1])
        indices_8bit = tf.reshape(tf.where(mask_8bit), [-1])

        # no compress
        v_32bit = tf.gather(tensor_flatten, indices_32bit)

        # 16bit compress
        s_16bit = tf.gather(sign, indices_16bit)
        e_16bit = tf.gather(exp, indices_16bit)
        m_16bit = tf.gather(mantissa, indices_16bit)
        n_shift = 127 - tf.cast(e_16bit, dtype=tf.int32)
        n_shift = tf.cast(n_shift, tf.uint32)
        shifted_s = tf.bitwise.right_shift(s_16bit, 8)
        marker = 0b00000000010000000000000000000000
        m_16bit_concat = tf.bitwise.bitwise_or(tf.bitwise.right_shift(m_16bit, 1), marker)
        shifted_m = tf.bitwise.right_shift(m_16bit_concat, n_shift)
        temp = tf.bitwise.bitwise_or(shifted_s, shifted_m)
        v_16bit = tf.cast(tf.bitwise.right_shift(temp, 8), dtype=tf.uint16)

        # 8bit compress
        s_8bit = tf.gather(sign, indices_8bit)
        e_8bit = tf.gather(exp, indices_8bit)
        m_8bit = tf.gather(mantissa, indices_8bit)
        n_shift = 127 - tf.cast(e_8bit, dtype=tf.int32)
        n_shift = tf.cast(n_shift, tf.uint32)
        shifted_s = tf.bitwise.right_shift(s_8bit, 8)
        marker = 0b00000000010000000000000000000000
        m_8bit_concat = tf.bitwise.bitwise_or(tf.bitwise.right_shift(m_8bit, 1), marker)
        shifted_m = tf.bitwise.right_shift(m_8bit_concat, n_shift)
        temp = tf.bitwise.bitwise_or(shifted_s, shifted_m)
        v_8bit = tf.cast(tf.bitwise.right_shift(temp, 16), dtype=tf.uint8)

        def encode_byte(a):
            # input: int32 type tensor with values in range 0,1,2,3 (2'b00,2'b01,2'b10,3'b11)
            # output: encoded uint8 type tensor
            a = tf.reshape(a, [-1])
            pad_size = 4 - tf.mod(tf.size(a), 4)
            pad = tf.range(0, pad_size)
            a = tf.concat([a, pad], 0)
            a_split1, a_split2, a_split3, a_split4 = tf.split(a, 4)

            # encode 4 grads into 1 Byte
            sum_1 = tf.add(a_split1, a_split2 * 4)
            sum_2 = tf.add(a_split3 * 16, a_split4 * 64)
            sum_all = tf.add(sum_1, sum_2)
            return tf.cast(sum_all, tf.uint8)

        # encode indices
        mask_encode = 0
        for mask, code in zip([mask_8bit, mask_16bit, mask_32bit], [1, 2, 3]):
            mask_encode += tf.cast(mask, tf.int32) * code
        mask_encode = encode_byte(mask_encode)
        v_32bit = tf.reshape(v_32bit, [-1])
        v_16bit = tf.reshape(v_16bit, [-1])
        v_8bit = tf.reshape(v_8bit, [-1])
        mask_encode = tf.reshape(mask_encode, [-1])
        tensor_compressed = v_32bit, v_16bit, v_8bit, mask_encode
        ctx = tensor_shape

        return tensor_compressed, ctx

    # decompress
    def decompress(self, tensor_compressed, ctx):

        def decode_byte(encoded, real_size):
            # input: encoded uint8 type tensor
            # output: int32 type tensor with values in range 0,1,2,3 (2'b00,2'b01,2'b10,3'b11)
            a = tf.cast(encoded, tf.int32)
            a_split1 = tf.mod(a, 4)
            a_split2 = tf.cast(tf.mod(a / 4, 4), tf.int32)
            a_split3 = tf.cast(tf.mod(a / 16, 4), tf.int32)
            a_split4 = tf.cast(tf.mod(a / 64, 4), tf.int32)
            a = tf.concat([a_split1, a_split2, a_split3, a_split4], 0)
            a = a[:real_size]
            return a

        v_32bit, v_16bit, v_8bit, mask_encode = tensor_compressed
        tensor_shape = ctx
        tensor_size = tf.math.reduce_prod(tensor_shape)

        # decode mask and gather indices
        mask_decode = decode_byte(mask_encode, tensor_size)
        mask_32bit = tf.equal(mask_decode, 3)
        mask_16bit = tf.equal(mask_decode, 2)
        mask_8bit = tf.equal(mask_decode, 1)
        indices_32bit = tf.reshape(tf.where(mask_32bit), [-1])
        indices_16bit = tf.reshape(tf.where(mask_16bit), [-1])
        indices_8bit = tf.reshape(tf.where(mask_8bit), [-1])

        edges_16bit = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        edges_8bit = [0, 2, 4, 8, 16, 32, 64, 128, 256]
        import tensorflow_probability as tfp

        # 16bit decompress
        # get the sign bit s_16bit and remove MSB from v_16bit
        s_16bit = tf.bitwise.bitwise_and(v_16bit, 0b1000000000000000)
        s_16bit = tf.cast(s_16bit, dtype=tf.int32)
        s_16bit = tf.bitwise.left_shift(s_16bit, 16)
        v_16bit = tf.bitwise.left_shift(v_16bit, 1)

        # 8bit decompress
        # get the sign bit s_8bit and remove MSB from v_8bit
        s_8bit = tf.bitwise.bitwise_and(v_8bit, 0b10000000)
        s_8bit = tf.cast(s_8bit, dtype=tf.int32)
        s_8bit = tf.bitwise.left_shift(s_8bit, 24)
        v_8bit = tf.bitwise.left_shift(v_8bit, 1)

        # find the marker bit in v_16bit and get the exponent
        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.int32), trainable=False)
        op = zero_tensor.assign(tf.zeros([tensor_size], dtype=tf.int32))
        with tf.control_dependencies([op]):
            temp = tf.scatter_update(zero_tensor, indices_16bit, tf.cast(v_16bit, tf.int32))
            temp = tf.scatter_update(temp, indices_8bit, tf.cast(v_8bit, tf.int32))
            n_shift_all = tfp.stats.find_bins(tf.cast(temp, dtype=tf.int32), edges_16bit)

        n_shift = 16 - tf.gather(n_shift_all, indices_16bit)
        e_16bit = 127 - (n_shift - 1)
        e_16bit = tf.bitwise.left_shift(e_16bit, 23)

        # restore the mantissa
        n_shift = tf.cast(n_shift, dtype=tf.uint16)
        v_16bit = tf.bitwise.left_shift(v_16bit, n_shift)
        v_16bit = tf.cast(v_16bit, dtype=tf.int32)
        m_16bit = tf.bitwise.left_shift(v_16bit, 7)

        # concat all
        temp = tf.bitwise.bitwise_or(s_16bit, e_16bit)
        v_16bit = tf.bitwise.bitwise_or(temp, m_16bit)
        v_16bit = tf.bitcast(v_16bit, tf.float32)

        # find the marker bit in v_8bit and get the exponent
        n_shift = 8 - tf.gather(n_shift_all, indices_8bit)
        e_8bit = 127 - (n_shift - 1)
        e_8bit = tf.bitwise.left_shift(e_8bit, 23)

        # restore the mantissa
        n_shift = tf.cast(n_shift, dtype=tf.uint8)
        v_8bit = tf.bitwise.left_shift(v_8bit, n_shift)
        v_8bit = tf.cast(v_8bit, dtype=tf.int32)
        m_8bit = tf.bitwise.left_shift(v_8bit, 15)

        # concat all
        temp = tf.bitwise.bitwise_or(s_8bit, e_8bit)
        v_8bit = tf.bitwise.bitwise_or(temp, m_8bit)
        v_8bit = tf.bitcast(v_8bit, tf.float32)

        zero_tensor = tf.Variable(tf.zeros([tensor_size], dtype=tf.float32), trainable=False)
        op = zero_tensor.assign(tf.zeros([tensor_size], dtype=tf.float32))
        with tf.control_dependencies([op]):
            temp = tf.scatter_update(zero_tensor, indices_32bit, v_32bit)
            temp = tf.scatter_update(temp, indices_16bit, v_16bit)
            temp = tf.scatter_update(temp, indices_8bit, v_8bit)
        tensor_decompressed = tf.reshape(temp, tensor_shape)
        return tensor_decompressed
