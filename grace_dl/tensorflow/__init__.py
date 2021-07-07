from abc import ABC, abstractmethod

import tensorflow as tf


class Memory(ABC):
    """
    Interface for implementing a memory operator
    """
    @staticmethod
    def init_var(initial_value):
        from collections import defaultdict
        residuals_init = defaultdict(list)
        for var in tf.trainable_variables():
            shape_str = ''.join(str(e) for e in var.get_shape().as_list())
            residuals_init[shape_str].append(tf.Variable(tf.zeros_like(var), trainable=False))

    @abstractmethod
    def compensate(self, tensor, name):
        """
        Compensate the tensor with the residuals.

        :param name: a unique id for this tensor
        :param tensor: the current state of the tensor
        :return: tensor compensated with stored values
        """
        raise NotImplemented("compensate was not implemented.")

    @abstractmethod
    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """
        Update the residuals.

        :param tensor: the original tensor
        :param name: a unique id for this tensor
        :param compressor: the compressor used to compress `tensor`
        :param tensor_compressed: a collection of tensors result of `compressor.compress`
        :param ctx: the context of compression
        :return: a collection of memory operations to be executed
        """
        raise NotImplemented("update was not implemented.")


class Compressor(ABC):
    """
    Interface for compressing and decompressing a given tensor.
    """

    def __init__(self, average=True, tensors_size_are_same=True):
        self.average = average
        self.tensors_size_are_same = tensors_size_are_same

    @abstractmethod
    def compress(self, tensor, name):
        """
        Compresses a tensor and returns a collection of compressed tensors with the context needed to decompress it.

        :param tensor: a tensor to compress
        :return: a collection of tensors and the context to decompress them
        """
        raise NotImplemented("compress was not implemented.")

    @abstractmethod
    def decompress(self, tensors, ctx):
        """
        Decompress a collection of compressed tensors with the given context.

        :param tensors: a collection of tensors
        :param ctx: context of the compression
        """
        raise NotImplemented("decompress was not implemented.")

    def aggregate(self, tensors):
        """
        Aggregate a collection of tensors.

        :param tensors: collection of tensors
        :return: aggregated tensor
        """
        return tf.math.add_n(tensors)


class Communicator(ABC):
    @abstractmethod
    def send_receive(self, tensors, ctx):
        """
        This method should use the collective communication, aggregation, and decompression methods.

        :param tensors: list of tensors to be sent
        :param ctx: compression context of the tensors
        :return: a decompressed tensor, result of aggregating the tensors of each worker
        """
        raise NotImplemented("send was not implemented.")

    def __init__(self, compressor, memory):
        """
        :param compressor: provides compression and decompression operators
        :param memory: provides compensation and update functions
        """
        self.compressor = compressor
        self.memory = memory

    def step(self, tensor, name):
        """
        Compensate, compress, update, communicate, decompress, and aggregate tensors of all the workers

        :param tensor: the tensor to be communicated across workers
        :return: the tensor after
        """
        tensor = self.memory.compensate(tensor, name)
        tensors_compressed, ctx = self.compressor.compress(tensor, name)
        update_ops = self.memory.update(tensor, name, self.compressor, tensors_compressed, ctx)
        with tf.control_dependencies(update_ops):
            return self.send_receive(tensors_compressed, ctx)
