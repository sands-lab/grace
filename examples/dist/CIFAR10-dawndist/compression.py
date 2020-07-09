# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gradient compression algorithms."""

import torch
from torch import distributed as dist


class Memory(object):
    def memory_compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        raise NotImplemented("memory_compensate was not implemented.")

    def memory_update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass


class NoneMemory(Memory):
    def memory_compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        return tensor

    def memory_update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        pass


class ResidualMemory(Memory):
    def __init__(self, beta=1.0, gamma=1.0):
        self.residuals = {}
        self.beta = beta
        self.gamma = gamma

    def memory_compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        if name in self.residuals:
            tensor = self.beta * self.residuals[name] + self.gamma * tensor
        return tensor

    def memory_update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        tensor_decompressed = compressor.decompress(tensor_compressed, ctx)
        residual = tensor - tensor_decompressed
        self.residuals[name] = residual


# class SimpleResidualMemory(Memory):
#     def __init__(self):
#         self.residuals = {}
#
#     def memory_compensate(self, tensor, name):
#         """Update the tensor with the residuals."""
#         if name in self.residuals:
#             tensor.add_(self.residuals[name])
#         return tensor
#
#     def memory_update(self, tensor, name, compressor, tensor_compressed, ctx):
#         """Update the residuals."""
#         indices, numel, shape = ctx
#         tensor[indices] = 0
#         self.residuals[name] = tensor


class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self, memory, average=True, tensors_size_are_same=True):
        self.memory = memory
        self.average = average
        self.tensors_size_are_same = tensors_size_are_same

    def compress(self, tensor, name):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compress was not implemented.")

    def decompress(self, tensors, ctx):
        """Decompress the tensor with the given context."""
        raise NotImplemented("decompress was not implemented.")

    def memory_compensate(self, tensor, name):
        return self.memory.memory_compensate(tensor, name)

    def memory_update(self, tensor, name, tensor_compressed, ctx):
        self.memory.memory_update(tensor, name, self, tensor_compressed, ctx)

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        return sum(tensors)


class NoneCompressor(Compressor):
    """Default no-op compression."""

    def __init__(self):
        super().__init__(memory=NoneMemory())

    def compress(self, tensor, name):
        return [tensor], None

    def decompress(self, tensors, ctx):
        tensor, = tensors
        return tensor


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""

    def __init__(self, memory=NoneMemory()):
        super().__init__(memory)

    def compress(self, tensor, name):
        """Downcasts the tensor to 16-bit."""
        dtype = tensor.dtype
        if dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor = tensor.type(torch.float16)
        return [tensor], dtype

    def decompress(self, tensors, dtype):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed, = tensors
        if dtype.is_floating_point:
            tensor_decompressed = tensor_decompressed.type(dtype)
        return tensor_decompressed


class RandomKCompressor(Compressor):
    """Python libraries Based Compress by performing sparsification (i.e., sending a ratio of the actual tensor size."""

    def __init__(self, compress_ratio=0.3, memory=NoneMemory()):
        super().__init__(memory)
        self.global_step = 0
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        """Use Python Random libraries RNG to compress by generating a list of indices to be transmitted."""
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        h = sum(bytes(name, encoding='utf8'), self.global_step)
        self.global_step += 1
        torch.manual_seed(h)
        k = max(1, int(numel * self.compress_ratio))

        indices = torch.randperm(numel, device=tensor.device)[:k]
        tensor_sparsed = tensor[indices]

        ctx = indices, numel, shape
        return [tensor_sparsed], ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        indices, numel, shape = ctx
        tensor, = tensors
        tensor_decompressed = torch.zeros(numel, dtype=tensor.dtype, layout=tensor.layout, device=tensor.device)
        tensor_decompressed.scatter_(0, indices, tensor)
        return tensor_decompressed.view(shape)


class TopKCompressor(Compressor):

    def __init__(self, compress_ratio=0.3, memory=NoneMemory()):
        super().__init__(memory)
        self.compress_ratio = compress_ratio

    def compress(self, tensor, params):
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        k = max(1, int(numel * self.compress_ratio))
        _, indices = torch.topk(tensor.abs(), k)
        tensor_sparsed = tensor[indices]
        ctx = numel, shape
        return [tensor_sparsed, indices], ctx

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        tensor, indices = tensors
        tensor_decompressed = torch.zeros(numel, dtype=tensor.dtype, layout=tensor.layout, device=tensor.device)
        tensor_decompressed.scatter_(0, indices, tensor)
        return tensor_decompressed.view(shape)


class ThresholdCompressor(Compressor):

    def __init__(self, threshold_val=256, memory=NoneMemory()):
        super().__init__(memory, tensors_size_are_same=False)
        self.threshold_val = threshold_val

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        indices, = torch.where(tensor.abs() >= self.threshold_val)
        values = tensor[indices]
        ctx = shape, numel
        return [values, indices], ctx

    def decompress(self, tensor_compressed, ctx):
        shape, numel = ctx
        values, indices = tensor_compressed
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)


class SignSGDCompressor(Compressor):

    def __init__(self, memory=NoneMemory()):
        super().__init__(memory=memory, average=False)

    def compress(self, tensor, name):
        """Encoding and compressing the signs """
        shape = tensor.size()
        tensor = tensor.flatten()

        tensor_flatten = tensor.flatten()
        tensor_compressed = tensor_flatten >= 0
        return [tensor_compressed.type(torch.uint8)], shape

    def decompress(self, tensors, shape):
        """Decoding the signs to float format """
        sign_encode, = tensors
        sign_decode = sign_encode.type(torch.float32) * 2 - 1
        tensor_decompressed = sign_decode.view(shape)
        return tensor_decompressed

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        agged_tensor = super().aggregate(tensors)
        agged_tensor = agged_tensor >= 0
        agged_tensor = agged_tensor * 2.0 - 1.0
        return agged_tensor


class EFSignSGDCompressor(Compressor):

    def __init__(self, lr=0.1):
        super().__init__(memory=ResidualMemory(beta=1, gamma=lr), average=False)
        self.learning_rate = lr

    def compress(self, tensor, name):
        """Encoding and compressing the signs """
        shape = tensor.size()
        tensor = tensor.flatten()

        sign_encode = tensor >= 0
        mean = tensor.abs().mean()
        tensor_compressed = mean, sign_encode.type(torch.uint8)

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        """Decoding the signs to float format """
        mean, sign_encode = tensor_compressed
        sign_decode = sign_encode.type(torch.float32) * 2 - 1
        sign_decode = mean * sign_decode
        tensor_decompressed = sign_decode.view(shape)
        return tensor_decompressed

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        agged_tensor = super().aggregate(tensors)
        agged_tensor = agged_tensor / self.learning_rate
        return agged_tensor


class SignumCompressor(Compressor):

    def __init__(self, momentum=0.9, memory=NoneMemory()):
        super().__init__(memory=memory, average=False)
        self.momentum = momentum
        self.momentums = {}

    def compress(self, tensor, name):
        """Encoding and compressing the signs """
        shape = tensor.size()
        tensor = tensor.flatten()

        # update tensor by momentum
        if name in self.momentums:
            tensor = (1.0 - self.momentum) * tensor + self.momentum * self.momentums[name]
        self.momentums[name] = tensor
        tensor_compressed = tensor >= 0
        return [tensor_compressed.type(torch.uint8)], shape

    def decompress(self, tensors, shape):
        sign_encode, = tensors
        """Decoding the signs to float format """
        sign_decode = sign_encode.type(torch.float32) * 2 - 1
        tensor_decompressed = sign_decode.view(shape)
        return tensor_decompressed

    def aggregate(self, tensors):
        """Aggregate a list of tensors."""
        agged_tensor = super().aggregate(tensors)
        agged_tensor = agged_tensor >= 0
        agged_tensor = agged_tensor * 2.0 - 1.0
        return agged_tensor


class QSGDCompressor(Compressor):

    def __init__(self, quantum_num=256, memory=NoneMemory()):
        super().__init__(memory)
        self.quantum_num = quantum_num

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        norm = tensor.norm()
        abs_gradient = tensor.abs()

        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = (previous_level + is_next_level)

        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(torch.int8 if self.quantum_num < 128 else torch.half)
        tensor_compressed = tensor_compressed, norm

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, norm = tensor_compressed

        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = norm / self.quantum_num * decode_output
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed


class OneBitCompressor(Compressor):

    def __init__(self, memory=NoneMemory()):
        super().__init__(memory)

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        mask0 = tensor < 0
        sum0 = torch.sum(tensor[mask0])
        num0 = torch.sum(mask0).float()
        mean0 = sum0 / num0 if num0 > 0 else sum0

        mask1 = ~mask0
        sum1 = torch.sum(tensor[mask1])
        num1 = numel - num0
        mean1 = sum1 / num1 if num1 > 0 else sum1

        tensor_compressed = mask0.type(torch.uint8), mean0, mean1

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        mask0, mean0, mean1 = tensor_compressed
        tensor_decompressed = mask0 * mean0 + ~mask0 * mean1
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed


class TernGradCompressor(Compressor):

    def __init__(self, memory=NoneMemory()):
        super().__init__(memory)

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()

        std = (tensor - torch.mean(tensor)) ** 2
        std = torch.sqrt(torch.mean(std))
        c = 2.5 * std.item()
        gradient = torch.clamp(tensor, -c, c)
        abs_gradient = gradient.abs()
        scalar = abs_gradient.max()

        sign_gradient = gradient.sign() * scalar
        rnd_sample = torch.empty_like(tensor).uniform_(0, scalar.item())
        sign_gradient[rnd_sample >= abs_gradient] = 0
        new_sign = sign_gradient.sign()  # -1, 0, 1

        tensor_compressed = new_sign.type(torch.int8), scalar.flatten()

        return tensor_compressed, shape

    def decompress(self, tensor_compressed, shape):
        tensor_compressed, scalar = tensor_compressed
        sign = tensor_compressed.type(torch.float32)
        tensor_decompressed = sign * scalar
        return tensor_decompressed.view(shape)


class DgcMemory(Memory):
    def __init__(self, momentum, gradient_clipping, world_size):
        self.gradient_clipping = gradient_clipping
        self.momentum = momentum
        self.world_size = world_size
        self.gradients = {}
        self.residuals = {}

    def memory_compensate(self, tensor, name):
        """Update the tensor with the residuals."""
        if self.gradient_clipping:
            tensor_squ_sum = torch.sum(tensor * tensor)
            clipping_val = torch.sqrt(dist.all_reduce(tensor_squ_sum) / self.world_size)
            tensor = tensor.clamp(-clipping_val, clipping_val)
        if name in self.residuals:
            self.residuals[name] = self.momentum * self.residuals[name] + tensor
        else:
            self.residuals[name] = tensor
        if name in self.gradients:
            self.gradients[name] += self.residuals[name]
            tensor = self.gradients[name]
        else:
            self.gradients[name] = tensor
        return tensor

    def memory_update(self, tensor, name, compressor, tensor_compressed, ctx):
        """Update the residuals."""
        shape, mask, _ = ctx
        not_mask = ~mask.view(shape)
        temp = self.residuals[name] * not_mask
        self.residuals[name] = temp
        temp = self.gradients[name] * not_mask
        self.gradients[name] = temp


class DgcCompressor(Compressor):

    def __init__(self, compress_ratio=0.3, world_size=1, momentum=0.9, gradient_clipping=False):
        super().__init__(memory=DgcMemory(momentum, gradient_clipping, world_size), tensors_size_are_same=False)
        self.compress_ratio = compress_ratio

    def compress(self, tensor, name):
        shape = tensor.size()
        tensor = tensor.flatten()
        numel = tensor.numel()

        sample_shape = [max(1, int(numel * 0.01))]
        sample_index = torch.empty(sample_shape).uniform_(0, numel).type(torch.long)
        sample_tensor = tensor[sample_index]

        k = max(1, int(numel * self.compress_ratio * 0.01))
        vals, indices = torch.topk(sample_tensor.abs(), k)

        thr = vals.min()
        mask = tensor.abs() >= thr
        selected = mask.sum()

        for _ in range(10):
            if selected > 1.3 * numel * self.compress_ratio:
                thr = 1.3 * thr
            elif selected < 0.7 * numel * self.compress_ratio:
                thr = 0.7 * thr
            else:
                break
            mask = tensor.abs() >= thr
            selected = mask.sum()

        indices, = torch.where(mask)
        values = tensor[indices]

        tensor_compressed = values, indices
        ctx = shape, mask, numel
        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx):
        values, indices = tensor_compressed
        shape, _, numel = ctx
        tensor_decompressed = torch.zeros(numel, dtype=values.dtype, layout=values.layout, device=values.device)
        tensor_decompressed.scatter_(0, indices, values)
        return tensor_decompressed.view(shape)


@torch.jit.script
def orthogonalize(matrix):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i: i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1:]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col


class PowerSGDCompressor(Compressor):

    def __init__(self, rank=2, use_memory=False, world_size=1):
        super().__init__(memory=NoneMemory())
        self.q_memory = {}
        self.rank = rank
        self.use_memory = use_memory
        self.world_size = world_size

    def compress(self, tensor, name):
        if tensor.dim() == 1:
            return [tensor], None

        shape = tensor.size()
        matrix = tensor.view([shape[0], -1])
        n, m = matrix.size()
        r = min(n, m, self.rank)
        if self.use_memory and name in self.q_memory:
            q = self.q_memory[name]
        else:
            q = torch.empty(m, r, dtype=matrix.dtype, layout=matrix.layout, device=matrix.device).normal_()
            # q, _ = torch.qr(q)
            orthogonalize(q)

        p = torch.mm(matrix, q)
        dist.all_reduce(p)
        p = p / self.world_size
        # p, _ = torch.qr(p)
        orthogonalize(p)
        q = torch.mm(matrix.t(), p)
        dist.all_reduce(q)
        q = q / self.world_size
        ctx = p, q, shape
        if self.use_memory:
            self.q_memory[name] = q
        return [], ctx

    def decompress(self, tensors, ctx):
        if ctx is None:
            tensor, = tensors
            return tensor
        p, q, tensor_shape = ctx
        new_tensor = torch.mm(p, q.t())
        tensor_decompressed = new_tensor.view(tensor_shape)
        return tensor_decompressed


class Compression(object):
    """Optional gradient compression algorithm used during allreduce."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor()

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor()
