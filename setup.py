"""Setup script for GPU Kernel Study."""

import glob

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 자동으로 csrc/ 디렉토리의 모든 .cu 파일 찾기
cuda_sources = glob.glob("csrc/*.cu")

# CUDA compute capability 설정 (A10 GPU: 8.6)
# 다른 GPU도 지원하도록 여러 아키텍처 포함
extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "-gencode=arch=compute_86,code=sm_86",  # A10, RTX 3090, A40
        "-gencode=arch=compute_80,code=sm_80",  # A100
        "-gencode=arch=compute_75,code=sm_75",  # T4, Titan RTX
        "--use_fast_math",
    ],
}

setup(
    name="gpu-kernel-study",
    version="0.1.0",
    description="CUDA & Triton kernel study",
    author="AIDA Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name="gpu_kernel_study.cuda_ops",
            sources=cuda_sources,
            include_dirs=["csrc"],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.11",
    # Dependencies are managed in pyproject.toml via uv
    install_requires=[],
    extras_require={},
    zip_safe=False,
)
