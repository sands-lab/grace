import tensorflow as tf

from grace_dl.tensorflow import Compressor


def sparsify(tensor, compress_ratio):
    elemnum = tensor.get_shape().as_list()[0]
    k = max(1, int(elemnum * compress_ratio))
    _, indices = tf.math.top_k(tf.math.abs(tensor), k)
    values = tf.gather(tensor, indices)
    return indices, values


def desparsify(indices, values, tensor_size):
    indices = tf.expand_dims(indices, 1)
    tensor = tf.scatter_nd(indices, values, [tensor_size])
    return tensor


class RandomKCompressor(Compressor):
    global_step = 0

    def __init__(self, compress_ratio):
        super().__init__()
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        """Use Python Random libraries RNG to compress by generating a list of indices to be transmitted."""

        tensor_shape = tf.shape(tensor)
        tensor_flatten = tf.reshape(tensor, [-1])
        elemnum = tensor_flatten.get_shape().as_list()[0]
        # all_indices = tf.range(elemnum, dtype=tf.int32)
        k = max(1, int(elemnum * self.compress_ratio))
        seed = (sum([ord(s) for s in tensor.name]), tf.train.get_global_step())
        tf.compat.v1.set_random_seed(1)
        indices = tf.random.stateless_uniform(
            [k], seed, minval=0, maxval=elemnum + 1, dtype=tf.int32, name=None
        )
        values = tf.gather(tensor_flatten, indices)
        ctx = indices, tensor_shape
        tensor_compressed = values

        return [tensor_compressed], ctx

    def decompress(self, tensor_compressed, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        indices, tensor_shape = ctx
        values, = tensor_compressed
        tensor_size = tf.math.reduce_prod(tensor_shape)
        tensor_decompressed = desparsify(indices, values, tensor_size)
        tensor_decompressed = tf.reshape(tensor_decompressed, tensor_shape)
        return tensor_decompressed
