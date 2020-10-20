from abc import ABC, abstractmethod


class Memory(ABC):
    @abstractmethod
    def compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        raise NotImplemented("compensate was not implemented.")

    def update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass


class Compressor(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self, average=True, tensors_size_are_same=True):
        self.average = average
        self.tensors_size_are_same = tensors_size_are_same

    @abstractmethod
    def compress(self, tensor, name):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compress was not implemented.")

    @abstractmethod
    def decompress(self, tensors, ctx):
        """Decompress the tensor with the given context."""
        raise NotImplemented("decompress was not implemented.")

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        return sum(tensors)


class Communicator(ABC):
    @abstractmethod
    def send_receive(self, tensors, name, ctx):
        raise NotImplemented("send was not implemented.")

    def __init__(self, compressor, memory, world_size):
        self.compressor = compressor
        self.memory = memory
        self.world_size = world_size

    def step(self, tensor, name):
        tensor = self.memory.compensate(tensor, name)
        tensors_compressed, ctx = self.compressor.compress(tensor, name)
        self.memory.update(tensor, name, self.compressor, tensors_compressed, ctx)
        return self.send_receive(tensors_compressed, name, ctx)
