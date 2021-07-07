import tensorflow as tf

from grace_dl.tensorflow import Memory
from horovod.tensorflow import _allreduce


class DgcMemory(Memory):
    def __init__(self, momentum, gradient_clipping, world_size):
        self.world_size = float(world_size)
        self.gradient_clipping = gradient_clipping
        self.momentum = momentum
        self.gradients = {}
        self.residuals = {}
        for v in tf.trainable_variables():
            self.residuals[v.name] = tf.Variable(tf.zeros_like(v), trainable=False)
            self.gradients[v.name] = tf.Variable(tf.zeros_like(v), trainable=False)

    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""

        if self.gradient_clipping:
            tensor_squ_sum = tf.math.reduce_sum(tf.math.square(tensor))
            thr_global = tf.math.sqrt(_allreduce(tensor_squ_sum))
            clipping_val = thr_global / tf.math.sqrt(self.world_size)
            tensor = tf.clip_by_value(tensor, -clipping_val, clipping_val)

        residual = self.residuals[name].assign(self.momentum * self.residuals[name] + tensor)
        tensor_compensate = self.gradients[name].assign(self.gradients[name] + residual) + tf.zeros_like(tensor)
        return tensor_compensate

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        _, mask = ctx
        not_mask = tf.cast(tf.math.logical_not(mask), tf.float32)
        not_mask = tf.reshape(not_mask, tf.shape(tensor))
        op1 = self.residuals[name].assign(self.residuals[name] * not_mask)
        op2 = self.gradients[name].assign(self.gradients[name] * not_mask)
        return [op1, op2]
