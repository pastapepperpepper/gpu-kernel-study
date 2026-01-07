"""Setup script for GPU Kernel Study."""

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gpu-kernel-study",
    version="0.1.0",
    description="CUDA & Triton kernel study",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            "gpu_kernel_study.cuda_ops",
            [
                "csrc/vector_add.cu",
                # Add new kernel files here when adding new kernels
            ],
            include_dirs=["csrc"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.11,<3.14",
    install_requires=[
        "torch==2.7.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "triton": ["triton>=3.0.0,<4.0.0"],
    },
)
