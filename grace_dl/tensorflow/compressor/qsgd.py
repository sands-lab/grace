import tensorflow as tf

from grace_dl.tensorflow import Compressor


class QSGDCompressor(Compressor):
    """
    (2016). QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding.
    Retrieved from http://arxiv.org/abs/1610.02132
    """
    def __init__(self, quantum_num):
        super().__init__()
        self.quantum_num = quantum_num

    def compress(self, tensor, name):
        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        norm = tf.reshape(tf.norm(tensor_flatten), [-1])
        abs_gradient = tf.abs(tensor_flatten)
        qnum = tf.cast(self.quantum_num, dtype=tf.float32)

        level_float = qnum / norm * abs_gradient
        previous_level = tf.math.floor(level_float)
        prob = tf.random.uniform(tf.shape(tensor_flatten))
        is_next_level = tf.cast(tf.math.less(prob, (level_float - previous_level)), tf.float32)
        new_level = tf.cast(previous_level + is_next_level, tf.float32)

        sign = tf.sign(tensor_flatten)
        tensor_compressed = new_level * sign
        tensor_compressed = tf.cast(tensor_compressed, dtype=tf.int8 if self.quantum_num < 128 else tf.int16)
        tensor_compressed = tensor_compressed, norm
        ctx = tensor_shape

        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx):
        qnum = tf.cast(self.quantum_num, dtype=tf.float32)
        tensor_shape = ctx
        tensor_compressed, norm = tensor_compressed

        decode_output = tf.cast(tensor_compressed, dtype=tf.float32)
        tensor_decompressed = norm / qnum * decode_output
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed
