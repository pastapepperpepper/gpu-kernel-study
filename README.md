# gpu-kernel-study

Repository for CUDA & Triton kernel study

## 프로젝트 구조

```
gpu-kernel-study/
├── csrc/                          # CUDA 커널 소스
│   └── vector_add.cu             # CUDA 커널 + Python 바인딩
│
├── src/
│   └── gpu_kernel_study/         # Python 패키지
│       ├── __init__.py
│       ├── cuda_ops.pyi          # 타입 스텁 (IDE 자동완성)
│       ├── kernels/              # 통합 Python 인터페이스
│       │   ├── __init__.py
│       │   └── vector_add.py     # CUDA + Triton wrapper
│       └── triton/               # Triton 커널 구현
│           ├── __init__.py
│           └── vector_add.py
│
├── tests/                         # 테스트 코드
│   └── test_vector_add.py
│
├── setup.py                       # Python extension 빌드
├── pyproject.toml
└── README.md
```

## 요구사항

- **Python** >= 3.11, < 3.14
- **CUDA Toolkit** >= 12.6 (권장)
- **PyTorch** 2.7.0
- **Triton** >= 3.0.0 (선택, 성능 비교용)

## 환경 세팅 (최초 1회)

```bash
# 의존성 설치
uv sync

# 가상환경 접속
source .venv/bin/activate

# 또는 pip 사용
uv pip install -e . --no-build-isolation

# 가상환경 내가 아닌 시스템에서 pre-commit 설치
pip install pre-commit
```

## 매일 개발 시

```bash
# 터미널 열고 활성화
source .venv/bin/activate

# 코드 수정
# Python 파일(.py): 수정 후 바로 실행
# CUDA 파일(.cu): 수정 후 아래 명령어로 빌드
uv pip install -e . --no-build-isolation

# 테스트/실행
pytest tests/
```


## 사용 방법

### 기본 사용

```python
import torch
from gpu_kernel_study.kernels import vector_add

# CUDA 텐서 생성
a = torch.ones(1024, device='cuda', dtype=torch.float32)
b = torch.ones(1024, device='cuda', dtype=torch.float32)

# CUDA 백엔드 사용
result_cuda = vector_add(a, b, backend='cuda')

# Triton 백엔드 사용
result_triton = vector_add(a, b, backend='triton')
```

### 성능 비교

```python
from gpu_kernel_study.kernels.vector_add import benchmark

a = torch.ones(1024, device='cuda')
b = torch.ones(1024, device='cuda')

# CUDA vs Triton 성능 비교
results = benchmark(a, b, num_runs=100)
```

### 테스트 실행

```bash
pytest tests/
```

## 새 커널 추가하기

1. **CUDA 커널 추가**: `csrc/new_kernel.cu` 작성
2. **Triton 커널 추가**: `src/gpu_kernel_study/triton/new_kernel.py` 작성
3. **Python Wrapper 추가**: `src/gpu_kernel_study/kernels/new_kernel.py` 작성
4. **setup.py 업데이트**: `csrc/new_kernel.cu` 추가
5. **재빌드**: `pip install -e .`

## 설정

- **uv**: Python 패키지 관리 (`pyproject.toml`)
- **pre-commit**: 커밋 전 스타일 점검
- **clang-format**: CUDA/C++ 코드 포매팅
