import torch
# from torch.utils.cpp_extension import load
# rdxtopk = load(name="rdxtopk", sources=["rdxtopk.cpp", "rdxtopk_cuda.cu"], verbose=True)
import qsgd_cuda

t = torch.rand(200000, dtype=torch.float32, device='cuda')
print(t)

out, scaler = qsgd_cuda.compress(t, 127, 128)
print(out, scaler)

out2 = qsgd_cuda.decompress(out, scaler, 127, 128)
print(out2)