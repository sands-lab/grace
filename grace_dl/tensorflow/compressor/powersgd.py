import tensorflow as tf

from grace_dl.tensorflow import Compressor
from horovod.tensorflow import _allreduce


class PowerSGDCompressor(Compressor):
    """
    PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization.
    T. Vogels, S. P. Karimireddy, and M. Jaggi. In NeurIPS, 2019.
    """

    def __init__(self, momentum_factor, world_size):
        super().__init__()
        self.q_memory = {}
        self.world_size = world_size
        self.momentum = {}
        self.momentum_factor = momentum_factor
        for v in tf.trainable_variables():
            self.momentum[v.name] = tf.Variable(tf.zeros_like(v), trainable=False)

    def compress(self, tensor, name):
        tensor_dims = len(tensor.get_shape().as_list())
        if tensor_dims == 1:
            return [tensor], None

        tensor_shape = tf.shape(tensor)
        matrix = tf.reshape(tensor, [tensor_shape[0], -1])

        q = self.q_memory[name]
        q, _ = tf.linalg.qr(q)

        p = tf.linalg.matmul(matrix, q)
        p = _allreduce(p) / self.world_size
        p, _ = tf.linalg.qr(p)
        q = tf.linalg.matmul(matrix, p, transpose_a=True)
        q = _allreduce(q) / self.world_size
        new_q = self.q_memory[name].assign(q)
        ctx = p, new_q, tensor_shape, name
        # variable initialization needs to be called before communication starts
        # self.momentum[tensor.name] = tf.Variable(tf.zeros_like(tensor), trainable=False)

        return [], ctx

    def decompress(self, tensors, ctx):
        if ctx is None:
            tensor, = tensors
            return tensor

        p, q, tensor_shape, tensor_name = ctx
        new_tensor = tf.linalg.matmul(p, q, transpose_b=True)
        new_tensor = tf.reshape(new_tensor, tensor_shape)

        new_momentum = self.momentum[tensor_name].assign(self.momentum_factor * self.momentum[tensor_name] +
                                                         (1 - self.momentum_factor) * new_tensor)
        tensor_decompressed = new_momentum + new_tensor

        return tensor_decompressed
