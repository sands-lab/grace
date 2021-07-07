import torch
# from torch.utils.cpp_extension import load
# rdxtopk = load(name="rdxtopk", sources=["rdxtopk.cpp", "rdxtopk_cuda.cu"], verbose=True)
import rdxtopk

size = 500000
k = int(size * 0.01)
t1 = torch.randint(100000, [size], device='cuda').float() + torch.rand(size, dtype=torch.float32, device='cuda')
# t2 = torch.tensor([11.0, 0.1, 22.0, 33.0, 44.0, 0.1, 55.0, 66.0, 77.0,], dtype=torch.float32, device='cuda')
t = torch.cat([t1,], dim=0)

indices = torch.arange(t.numel()).cuda().int()

print("===== original tensor ======")
print(t)

print("\n\n===== Torch TopK ======")
out3, out4 = torch.topk(t, k)
print(out3, out4)

print("\n\n===== RadixTopK ======")
out1, out2 = rdxtopk.topk(t, indices, k)

print("\n\n===== value is sorted ======")
print(out1.sort(descending=True)[0], out2[out1.sort(descending=True)[1]])

print("\n\n===== value is not sorted ======")
print(out1, out2)

# a = torch.zeros(size, device='cuda').float()
# b = torch.zeros(size, device='cuda').float()
#
# a[out4.long()] = out3
# b[out2.long()] = out1
# print(torch.sum(a), torch.sum(b), torch.sum((a - b).abs()))


