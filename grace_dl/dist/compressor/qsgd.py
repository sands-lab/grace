import torch
from grace_dl.dist import Compressor


class QSGDCompressor(Compressor):

    def __init__(self, quantum_num, bucket_size=128):
        super().__init__()
        self.quantum_num = quantum_num
        self.bucket_size = bucket_size

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()
        abs_gradient = tensor.abs()

        if tensor.numel() % self.bucket_size != 0:
            pad_size = self.bucket_size - tensor.numel() % self.bucket_size
            pad_tensor = torch.cat([tensor, torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)])
        else:
            pad_tensor = tensor
        pad_tensor = pad_tensor.view([-1, self.bucket_size])
        pad_tensor_sqsum = torch.sum(pad_tensor ** 2, dim=1)
        bucket_norm = torch.sqrt(pad_tensor_sqsum)
        b = torch.ones([1, self.bucket_size], device=tensor.device)
        expand_norm = torch.matmul(bucket_norm.view([-1, 1]), b)
        norm = expand_norm.flatten()[:tensor.numel()]

        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = (previous_level + is_next_level)

        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(torch.int8 if self.quantum_num < 128 else torch.half)

        return (tensor_compressed, bucket_norm), shape

    def decompress(self, tensor_compressed, ctx):
        tensor_compressed, bucket_norm = tensor_compressed
        shape = ctx
        b = torch.ones([1, self.bucket_size], device=tensor_compressed.device)
        expand_norm = torch.matmul(bucket_norm.view([-1, 1]), b)
        norm = expand_norm.flatten()[:shape.numel()]
        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = norm / self.quantum_num * decode_output
        tensor_decompressed = tensor_decompressed.view(shape)

        return tensor_decompressed


class QSGDCompressor_CUDA(Compressor):

    def __init__(self, quantum_num, bucket_size=128):
        super().__init__()
        self.quantum_num = quantum_num
        self.bucket_size = bucket_size

    def compress(self, tensor, name):
        import qsgd_cuda
        shape = tensor.size()
        tensor = tensor.flatten()

        tensor_compressed, bucket_norm = qsgd_cuda.compress(tensor, self.quantum_num, self.bucket_size)
        tensor_compressed = tensor_compressed, bucket_norm.float()

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        import qsgd_cuda
        tensor_compressed, bucket_norm = tensor_compressed

        tensor_decompressed = qsgd_cuda.decompress(tensor_compressed, bucket_norm.double(), self.quantum_num, self.bucket_size)

        return tensor_decompressed.view(shape)


