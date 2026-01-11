#!/usr/bin/env python3
"""환경 체크 스크립트 - CUDA, PyTorch, GPU 정보 확인"""

import subprocess
import sys


def check_python():
    """Python 버전 확인"""
    version = sys.version_info
    print(f"✓ Python 버전: {version.major}.{version.minor}.{version.micro}")
    if not (3, 11) <= (version.major, version.minor) < (3, 14):
        print("  ⚠️  경고: Python 3.11-3.13 권장")
    return f"{version.major}.{version.minor}"


def check_cuda():
    """CUDA 버전 확인"""
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=True
        )
        for line in result.stdout.split("\n"):
            if "release" in line.lower():
                print(f"✓ CUDA Compiler: {line.strip()}")
                # Extract version (e.g., "11.8" from "release 11.8")
                if "release" in line:
                    version = line.split("release")[1].strip().split(",")[0].strip()
                    return version
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ nvcc를 찾을 수 없습니다. CUDA Toolkit이 설치되어 있나요?")
        return None


def check_nvidia_smi():
    """nvidia-smi로 GPU 및 드라이버 정보 확인"""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        )
        print("\n=== GPU 정보 (nvidia-smi) ===")
        lines = result.stdout.split("\n")
        for _, line in enumerate(lines):
            # Driver version과 CUDA version 출력
            if "Driver Version" in line:
                print(f"✓ {line.strip()}")
            # GPU 이름 출력
            if "NVIDIA" in line and "|" in line:
                print(f"✓ GPU: {line.strip()}")
                # A10 확인
                if "A10" in line:
                    print("  ✓ A10 GPU 확인됨")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ nvidia-smi를 찾을 수 없습니다.")


def check_torch():
    """PyTorch 및 CUDA 지원 확인"""
    try:
        import torch

        print("\n=== PyTorch 정보 ===")
        print(f"✓ PyTorch 버전: {torch.__version__}")
        print(f"✓ CUDA 사용 가능: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"✓ CUDA 버전 (PyTorch 빌드): {torch.version.cuda}")
            print(f"✓ cuDNN 버전: {torch.backends.cudnn.version()}")
            print(f"✓ 사용 가능한 GPU 개수: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n  GPU {i}: {props.name}")
                print(f"    - Compute Capability: {props.major}.{props.minor}")
                print(f"    - Total Memory: {props.total_memory / 1024**3:.2f} GB")
        else:
            print("✗ CUDA를 사용할 수 없습니다!")

        return (
            torch.__version__,
            torch.version.cuda if torch.cuda.is_available() else None,
        )
    except ImportError:
        print("✗ PyTorch가 설치되어 있지 않습니다.")
        return None, None


def check_triton():
    """Triton 설치 확인"""
    try:
        import triton

        print("\n=== Triton 정보 ===")
        print(f"✓ Triton 버전: {triton.__version__}")
        return triton.__version__
    except ImportError:
        print("\n=== Triton 정보 ===")
        print("✗ Triton이 설치되어 있지 않습니다. (선택 사항)")
        return None


def check_build_extension():
    """CUDA extension 빌드 확인"""
    try:
        from gpu_kernel_study import cuda_ops

        print("\n=== CUDA Extension ===")
        print("✓ cuda_ops 모듈 로드 성공")

        # 간단한 테스트
        import torch

        if torch.cuda.is_available():
            a = torch.ones(10, device="cuda", dtype=torch.float32)
            b = torch.ones(10, device="cuda", dtype=torch.float32)
            c = cuda_ops.vector_add(a, b)
            if torch.allclose(c, torch.full((10,), 2.0, device="cuda")):
                print("✓ CUDA 커널 동작 확인 완료")
            else:
                print("✗ CUDA 커널 결과가 예상과 다릅니다.")
    except ImportError:
        print("\n=== CUDA Extension ===")
        print("✗ cuda_ops 모듈을 찾을 수 없습니다.")
        print("  빌드가 필요합니다: pip install -e .")
    except Exception as e:
        print(f"✗ CUDA 커널 실행 중 오류: {e}")


def main():
    print("=" * 60)
    print("GPU Kernel Study 환경 체크")
    print("=" * 60)

    py_ver = check_python()
    cuda_ver = check_cuda()
    check_nvidia_smi()
    torch_ver, torch_cuda = check_torch()
    triton_ver = check_triton()
    check_build_extension()

    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    print(f"Python: {py_ver}")
    print(f"CUDA Toolkit: {cuda_ver or 'N/A'}")
    print(f"PyTorch: {torch_ver or 'N/A'} (CUDA {torch_cuda or 'N/A'})")
    print(f"Triton: {triton_ver or 'Not installed'}")

    if cuda_ver and torch_cuda:
        cuda_major = cuda_ver.split(".")[0]
        torch_cuda_major = torch_cuda.split(".")[0]
        if cuda_major != torch_cuda_major:
            print(
                f"\n⚠️  경고: CUDA Toolkit ({cuda_ver})과 "
                f"PyTorch CUDA 버전 ({torch_cuda})이 다릅니다."
            )
            print("   PyTorch는 자체 CUDA 라이브러리를 사용하므로 일반적으로 문제없지만,")
            print("   커스텀 CUDA extension 빌드 시 호환성 문제가 발생할 수 있습니다.")


if __name__ == "__main__":
    main()
