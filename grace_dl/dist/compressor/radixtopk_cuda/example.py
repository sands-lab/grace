# Compilation cmd: python setup.py install
# Alternative JIT compilation, refer to https://pytorch.org/tutorials/advanced/cpp_extension.html#jit-compiling-extensions
# from torch.utils.cpp_extension import load
# rdxtopk = load(name="rdxtopk", sources=["rdxtopk.cpp", "rdxtopk_cuda.cu"], verbose=True)

import torch
import rdxtopk


size = 500000
k = int(size * 0.01)
t = torch.randint(100000, [size], device='cuda').float() + torch.rand(size, dtype=torch.float32, device='cuda')

indices = torch.arange(t.numel()).cuda().int()

print("===== original tensor ======")
print(t)

print("\n\n===== RadixTopK values not sorted ======")
out1, out2 = rdxtopk.topk(t, indices, k)
print(out1, out2)

print("\n\n===== RadixTopK values sorted ======")
print(out1.sort(descending=True)[0], out2[out1.sort(descending=True)[1]])

print("\n\n===== Torch TopK values sorted ======")
out3, out4 = torch.topk(t, k)
print(out3, out4)



