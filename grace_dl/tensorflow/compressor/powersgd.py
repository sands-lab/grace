import tensorflow as tf

from grace_dl.tensorflow import Compressor
from horovod.tensorflow import _allreduce


class PowerSGDCompressor(Compressor):
    """
    PowerSGD: Practical Low-Rank Gradient Compression for Distributed Optimization.
    T. Vogels, S. P. Karimireddy, and M. Jaggi. In NeurIPS, 2019.
    """

    def __init__(self, world_size):
        super().__init__()
        self.q_memory = {}
        self.world_size = world_size

    def compress(self, tensor):
        tensor_dims = len(tensor.get_shape().as_list())
        if tensor_dims == 1:
            return [tensor], None

        tensor_shape = tf.shape(tensor)
        matrix = tf.reshape(tensor, [tensor_shape[0], -1])

        q = self.q_memory[tensor.name]
        q, _ = tf.linalg.qr(q)

        p = tf.linalg.matmul(matrix, q)
        p = _allreduce(p) / self.world_size
        p, _ = tf.linalg.qr(p)
        q = tf.linalg.matmul(matrix, p, transpose_a=True)
        q = _allreduce(q) / self.world_size
        new_q = self.q_memory[tensor.name].assign(q)
        ctx = p, new_q, tensor_shape

        return [], ctx

    def decompress(self, tensors, ctx):
        if ctx is None:
            tensor, = tensors
            return tensor

        p, q, tensor_shape = ctx
        new_tensor = tf.linalg.matmul(p, q, transpose_b=True)
        new_tensor = tf.reshape(new_tensor, tensor_shape)

        return new_tensor
