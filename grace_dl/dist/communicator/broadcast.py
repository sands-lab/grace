import torch
from torch import distributed as dist

from grace_dl.dist import Communicator


class Broadcast(Communicator):

    def __init__(self, compressor, memory, world_size, rank):
        super().__init__(compressor, memory, world_size)
        self.rank = rank

    def send_receive(self, tensors, name, ctx):
        if not self.compressor.tensors_size_are_same:
            raise NotImplemented()

        tensors_decompressed = []
        for root_rank in range(self.world_size):
            if root_rank == self.rank:
                broadcasted = tensors
                for tensor_compressed in tensors:
                    dist.broadcast(tensor_compressed, root_rank)
            else:
                broadcasted = []
                for tensor_compressed in tensors:
                    tensor = torch.empty_like(tensor_compressed)
                    dist.broadcast(tensor, root_rank)
                    broadcasted.append(tensor)
            tensor_decompressed = self.compressor.decompress(broadcasted, ctx)
            tensors_decompressed.append(tensor_decompressed)

        tensor_aggregated = self.compressor.aggregate(tensors_decompressed)
        return (tensor_aggregated / self.world_size) if self.compressor.average else tensor_aggregated
