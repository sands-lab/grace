# conda install -c conda-forge cupy=7.0.0=py37h0c141eb_2
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack

from grace_dl.torch import Compressor


class NaturalCompressor(Compressor):
    def __init__(self):
        super().__init__()

    def compress(self, tensor, name):
        import cupy
        shape = tensor.size()
        tensor_flatten = tensor.flatten()
        cupy_tensor = cupy.fromDlpack(to_dlpack(tensor_flatten))
        tensor_cast = cupy_tensor.view(cupy.int32)
        sign = tensor_cast & cupy.int32(0b10000000000000000000000000000000)
        exp = tensor_cast & cupy.int32(0b01111111100000000000000000000000)
        mantissa = tensor_cast & cupy.int32(0b00000000011111111111111111111111)
        exp_add_one = mantissa > cupy.random.randint(low=0, high=0b00000000011111111111111111111111,
                                                     size=cupy_tensor.shape,
                                                     dtype=cupy.int32)
        exponent = cupy.where(exp_add_one, exp + 0b00000000100000000000000000000000, exp)
        exp_shift = cupy.clip(exponent, a_min=0b00001001000000000000000000000000, a_max=0b01001000100000000000000000000000)
        exps = cupy.right_shift(exp_shift, 23)
        exps = cupy.bitwise_or(cupy.right_shift(sign, 24), exps - 18)
        tensor_compressed = exps.astype(cupy.uint8)
        return [from_dlpack(tensor_compressed.toDlpack())], shape

    def decompress(self, tensor_compressed, shape):
        import cupy
        tensor_compressed, = tensor_compressed
        cupy_tensor = cupy.fromDlpack(to_dlpack(tensor_compressed))
        sign = cupy_tensor > 127
        exps = cupy.bitwise_and(cupy_tensor, 0b01111111)
        floats = cupy.left_shift((exps + 18).astype(cupy.int32), 23).view(cupy.float32)
        tensor_decompressed = cupy.where(sign, -floats, floats)
        tensor_decompressed = cupy.multiply((exps >= 1).astype(cupy.float32), tensor_decompressed)
        return from_dlpack(tensor_decompressed.toDlpack()).view(shape)


class NaturalCompressor_CUDA(Compressor):
    def __init__(self):
        super().__init__()

    def compress(self, tensor, name):
        import cnat_cuda
        shape = tensor.size()
        tensor_compressed = cnat_cuda.compress(tensor.flatten())
        return [tensor_compressed, ], shape

    def decompress(self, tensor_compressed, shape):
        import cnat_cuda
        tensor_compressed, = tensor_compressed
        tensor_decompressed = cnat_cuda.decompress(tensor_compressed)
        return tensor_decompressed.view(shape)