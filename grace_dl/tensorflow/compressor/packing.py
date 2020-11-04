import tensorflow as tf


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