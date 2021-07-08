# Compilation cmd: python setup.py install
# Alternative JIT compilation, refer to https://pytorch.org/tutorials/advanced/cpp_extension.html#jit-compiling-extensions
# from torch.utils.cpp_extension import load
# qsgd_cuda = load(name="qsgd_cuda", sources=["qsgd.cpp", "qsgd_cuda.cu"], verbose=True)

import torch
import qsgd_cuda

t = torch.rand(200000, dtype=torch.float32, device='cuda')
print(t)

out, scaler = qsgd_cuda.compress(t, 127, 128)
print(out, scaler)

out2 = qsgd_cuda.decompress(out, scaler, 127, 128)
print(out2)