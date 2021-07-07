import torch, math
from torch import distributed as dist

from grace_dl.dist import Communicator

from grace_dl.dist.compressor.qsgd import QSGDCompressor
from grace_dl.dist.compressor.qsgd import QSGDCompressor_CUDA
from grace_dl.dist.compressor.terngrad import TernGradCompressor
from grace_dl.dist.compressor.natural import NaturalCompressor
from grace_dl.dist.compressor.natural import NaturalCompressor_CUDA


class AllToAll(Communicator):

    def compress_allgather(self, tensors, ctx):
        tensors_gathered = []
        for tensor_compressed in tensors:
            tensor_list = [torch.empty_like(tensor_compressed) for _ in range(self.world_size)]
            dist.all_gather(tensor_list, tensor_compressed)
            tensors_gathered.append(tensor_list)

        decompressed_list = []
        for tensors_compressed in zip(*tensors_gathered):
            tensor_decompressed = self.compressor.decompress(tensors_compressed, ctx)
            decompressed_list.append(tensor_decompressed)
        tensors_concat = torch.cat(decompressed_list, dim=0)
        return tensors_concat

    def send_receive(self, tensors, name, ctx):

        if isinstance(self.compressor, QSGDCompressor) or isinstance(self.compressor, QSGDCompressor_CUDA):
            tensor, norm = tensors
            tensor_size = tensor.numel()
            world_size = self.world_size
            bucket_size = self.compressor.bucket_size

            pad_size = math.ceil(tensor.numel() / (world_size * bucket_size)) * (
                        world_size * bucket_size) - tensor.numel()
            padding = torch.empty(int(pad_size), dtype=tensor.dtype, layout=tensor.layout, device=tensor.device)
            tensor = torch.cat((tensor, padding), dim=0)

            pad_size = tensor.numel() / bucket_size - norm.numel()
            padding = torch.empty(int(pad_size), dtype=norm.dtype, layout=norm.layout, device=norm.device)
            norm = torch.cat((norm, padding), dim=0)

            data = tensor
            input = list(data.chunk(world_size))
            output = list(torch.empty([data.numel()], dtype=data.dtype, device=data.device).chunk(world_size))
            dist.all_to_all(output, input)
            tensor_chunk_list = output

            data = norm
            input = list(data.chunk(world_size))
            output = list(torch.empty([data.numel()], dtype=data.dtype, device=data.device).chunk(world_size))
            dist.all_to_all(output, input)
            norm_chunk_list = output

            compressed_list = [tensor_chunk_list, norm_chunk_list]
            decompressed_list = []
            for tensors_compressed in zip(*compressed_list):
                chunk_shape = tensors_compressed[0].size()
                tensor_decompressed = self.compressor.decompress(tensors_compressed, chunk_shape)
                decompressed_list.append(tensor_decompressed)
            tensors_aggregated = self.compressor.aggregate(decompressed_list)

        elif isinstance(self.compressor, TernGradCompressor):
            tensor, scalar = tensors
            tensor_size = tensor.numel()
            world_size = self.world_size

            pad_size = math.ceil(tensor.numel() / world_size) * world_size - tensor.numel()
            padding = torch.empty(int(pad_size), dtype=tensor.dtype, layout=tensor.layout, device=tensor.device)
            tensor = torch.cat((tensor, padding), dim=0)

            data = tensor
            input = list(data.chunk(world_size))
            output = list(torch.empty([data.numel()], dtype=data.dtype, device=data.device).chunk(world_size))
            dist.all_to_all(output, input)
            tensor_chunk_list = output

            input = [scalar for _ in range(world_size)]
            output = [torch.empty_like(scalar, device=scalar.device) for _ in range(world_size)]
            dist.all_to_all(output, input)
            scalar_chunk_list = output

            compressed_list = [tensor_chunk_list, scalar_chunk_list]
            decompressed_list = []
            for tensors_compressed in zip(*compressed_list):
                chunk_shape = tensors_compressed[0].size()
                tensor_decompressed = self.compressor.decompress(tensors_compressed, chunk_shape)
                decompressed_list.append(tensor_decompressed)
            tensors_aggregated = self.compressor.aggregate(decompressed_list)

        elif isinstance(self.compressor, NaturalCompressor) or isinstance(self.compressor, NaturalCompressor_CUDA):
            tensor = tensors[0]
            tensor_size = tensor.numel()
            world_size = self.world_size

            pad_size = math.ceil(tensor.numel() / world_size) * world_size - tensor.numel()
            padding = torch.empty(int(pad_size), dtype=tensor.dtype, layout=tensor.layout, device=tensor.device)
            tensor = torch.cat((tensor, padding), dim=0)

            data = tensor
            input = list(data.chunk(world_size))
            output = list(torch.empty([data.numel()], dtype=data.dtype, device=data.device).chunk(world_size))
            dist.all_to_all(output, input)
            tensor_chunk_list = output

            compressed_list = [[t, ] for t in tensor_chunk_list]
            decompressed_list = []
            for tensors_compressed in compressed_list:
                chunk_shape = tensors_compressed[0].size()
                tensor_decompressed = self.compressor.decompress(tensors_compressed, chunk_shape)
                decompressed_list.append(tensor_decompressed)
            tensors_aggregated = self.compressor.aggregate(decompressed_list)

        else:
            raise NotImplementedError(self.compressor)

        tensors_compressed, ctx2 = self.compressor.compress(tensors_aggregated, name)
        tensors_concat = self.compress_allgather(tensors_compressed, ctx2)
        tensors_aggregated = tensors_concat[:tensor_size].view(ctx)

        return (tensors_aggregated / self.world_size) if self.compressor.average else tensors_aggregated
