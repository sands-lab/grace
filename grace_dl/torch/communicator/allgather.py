import torch

from grace_dl.torch import Communicator
from horovod.torch import allgather, allgather_async, synchronize


class Allgather(Communicator):
    def __init__(self, compressor, memory, world_size):
        super().__init__(compressor, memory)
        self.world_size = world_size

    def async_send(self, tensors_compressed, name):
        """
        :param tensors_compressed: list of flat tensors to communicate
        :param name: for the all_gather operation
        :return: handles to synchronize, tensor sizes per rank
        """
        # list of tensor size for this rank, allgather on the 1st dimension of the tensor
        tensors_size = []
        for t in tensors_compressed:
            size_dim0 = t.size()[0] if len(t.size())>0 else t.numel()
            tensors_size.append(size_dim0)

        if self.compressor.tensors_size_are_same:
            tensors_size_ag = [tensors_size] * self.world_size  # list of tensor sizes per rank
            tensor_sizes = zip(*tensors_size_ag)  # transpose
        else:
            tensors_size = torch.tensor(tensors_size)  # TODO: set device
            gathered = allgather(tensors_size)  # tensor of tensor sizes per rank
            tensor_sizes = gathered.view([self.world_size, -1]).t().tolist()  # transpose, to list

        handles = []
        for tensor_compressed in tensors_compressed:
            handle = allgather_async(tensor_compressed)
            handles.append(handle)

        return handles, tensor_sizes

    def wait_receive(self, result, ctx):
        handles, tensor_sizes = result
        tensors_ag = []
        for handle, sizes in zip(handles, tensor_sizes):
            gathered = synchronize(handle)
            tensors_ag.append(gathered.split(sizes))
        # import time
        # time.sleep(0.001)
        list_tensor_decompressed = []
        for tensor_compressed in zip(*tensors_ag):
            tensor_decompressed = self.compressor.decompress(tensor_compressed, ctx)
            list_tensor_decompressed.append(tensor_decompressed)

        tensors_aggregated = self.compressor.aggregate(list_tensor_decompressed)
        return (tensors_aggregated / self.world_size) if self.compressor.average else tensors_aggregated
