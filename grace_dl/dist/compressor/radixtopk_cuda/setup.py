from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rdxtopk',
    ext_modules=[
        CUDAExtension('rdxtopk', [
            'rdxtopk.cpp',
            'rdxtopk_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
