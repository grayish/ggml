# 0. 관련 프로젝트 탐색과 비교

> 목표: "ggml을 vcpkg 친화적인 순수 C++23으로 다시 쓴다"는 아이디어가 이미 어디까지 시도되었는지 확인하고,
> 우리가 차지할 수 있는 자리를 찾는다. (조사 시점: 2026-09)

## 0.1 결론 먼저

- **ggml을 C++23으로 완전히 다시 쓴 공개 프로젝트는 찾지 못했다.** 검색되는 것은 ggml 포크(수백 개), C++ 래퍼(ggml-easy), 다른 언어 재구현(Rust candle, Zig 계열), 또는 ggml과 무관한 C++ 추론 엔진(gemma.cpp, MLX)뿐이다.
- **vcpkg에는 이미 `ggml`, `llama-cpp`, `whisper-cpp` 포트가 있다.** 2025-03-12에 머지된 [microsoft/vcpkg#43925](https://github.com/microsoft/vcpkg/pull/43925)로 추가되었고, `ggml` 포트는 blas/cuda/metal/opencl/openmp/vulkan 기능(feature)을 제공한다. 즉 "vcpkg로 ggml을 쓰는 것"은 해결된 문제이고, 우리의 차별점은 **라이브러리 자체의 언어와 설계**여야 한다.
- ggml 상류(upstream)는 C11 + C++17을 **의도적으로 고정**하고 있다. `src/CMakeLists.txt:609`에 `target_compile_features(... c_std_11 cxx_std_17) # don't bump`라고 명시돼 있다. 따라서 C++23 채택은 상류에 기여할 수 없는 방향이고, 별도 프로젝트가 정당화된다.
- 상류 자체도 C에서 C++로 서서히 이동 중이다. `src/` 최상위에서 순수 C 파일은 `ggml.c`, `ggml-alloc.c`, `ggml-quants.c` 세 개뿐이고, 백엔드·스케줄러·GGUF·최적화기·스레딩은 이미 `.cpp`다(전체 소스 기준 `.c` 91개 vs C++/CUDA/ObjC 443개). 단, 공개 API는 여전히 `extern "C"` C ABI다.

## 0.2 비교 표

| 프로젝트 | 언어/표준 | ggml과의 관계 | 패키징 | 우리 프로젝트에 주는 시사점 |
|---|---|---|---|---|
| [ggml-org/ggml](https://github.com/ggml-org/ggml) (이 저장소) | C11 + C++17, C ABI | 원본 | CMake `find_package(ggml)`, pkg-config, vcpkg 포트 `ggml` | 재구현의 기준. 동작 동일성을 검증하는 **오라클**로 사용 |
| vcpkg `ggml` 포트 | - | 원본을 패치해 빌드 | `vcpkg install ggml[cuda,vulkan]` | portfile이 AVX/FMA/F16C 등 네이티브 최적화를 끄고 범용 바이너리를 만든다. 우리도 "네이티브 여부"를 feature로 노출해야 함 |
| vcpkg `llama-cpp` 포트 | C++17 | ggml 포트에 의존 | `vcpkg install llama-cpp[vulkan]` | 외부 ggml 의존을 강제하는 패치가 필요했다는 점이 교훈. 처음부터 **외부 소비 전제**로 설계해야 함 |
| [ngxson/ggml-easy](https://github.com/ngxson/ggml-easy) | C++ 헤더 온리 래퍼 | ggml을 git submodule로 포함 | 없음 (submodule) | 사용자 경험(safetensors 직접 로딩, 중간 텐서 디버그 출력)에서 참고. 다만 래퍼일 뿐 내부는 C |
| [ggml#1326 RFC](https://github.com/ggml-org/ggml/issues/1326) | C++ 헤더 제안 | `a.norm(eps)`, `w ^ x` 같은 PyTorch식 연산자 오버로딩 | - | 상류에서도 C++ 편의 API 수요가 확인됨. 하지만 아직 미해결 |
| `include/ggml-cpp.h` | C++ | 상류가 제공하는 유일한 C++ 지원: `unique_ptr` 삭제자 8종 | ggml에 포함 | 상류의 C++ 지원 수준이 "삭제자"에 그친다는 증거 |
| [google/gemma.cpp](https://github.com/google/gemma.cpp) | C++ (Highway SIMD) | 무관, 설계 영감(ggml, llama.c)만 공유 | CMake, Bazel | **포터블 SIMD를 Highway 한 벌로 해결**하고 런타임에 ISA를 고른다. ggml은 아키텍처별 커널을 손으로 쓴다. C++23에는 `std::simd`가 없으므로(C++26) 우리도 이 선택지를 검토해야 함 |
| [ml-explore/mlx](https://github.com/ml-explore/mlx) | C++ 코어 + Python | 무관, 경쟁 | CMake, pip | 지연 평가 배열 + 통합 메모리 + Metal. "그래프를 먼저 만들고 나중에 실행"이라는 점은 ggml과 같지만, 사용자 API가 NumPy 수준으로 높다 |
| [huggingface/candle](https://github.com/huggingface/candle) | Rust | GGUF/GGML 양자화 파일을 읽고 자체 커널로 실행 | cargo | "ggml 파일 포맷 호환 + 다른 언어 재구현"의 성공 사례. 우리 프로젝트의 가장 가까운 선례(단, Rust) |
| [ZML](https://zml.ai/) | Zig + OpenXLA/MLIR | 무관 | - | 컴파일러 스택 접근. 참고만 |
| llama2.c / [llama2.zig](https://github.com/cgbur/llama2.zig) | C / Zig | 무관, 교육용 | - | 700줄짜리 단일 파일 추론. 학습 초기의 "최소 목표"로 적합 |
| CTranslate2, ncnn, ExecuTorch (참고) | C++17 | 무관 | vcpkg 포트 존재(ncnn 등) | 대규모 C++ 추론 엔진의 패키징 방식 참고 |

## 0.3 각 프로젝트 상세

### ggml (원본)의 현재 위치
- 공개 헤더 API 개수: `include/ggml.h`에 `GGML_API` 선언 379개.
- 연산 종류: `enum ggml_op` 약 110개(`include/ggml.h:492`), 타입 43개(`GGML_TYPE_COUNT = 43`, `include/ggml.h:433`).
- 백엔드 18종이 `src/CMakeLists.txt:485-605`의 `ggml_add_backend(...)`로 등록된다 (CPU, BLAS, CANN, CUDA, ET, HIP, METAL, MUSA, RPC, VirtGPU, SYCL, Vulkan, WebGPU, zDNN, OpenCL, Hexagon, ZenDNN, OPENVINO).
- 기여 정책: 핵심 변경은 llama.cpp 저장소에서 PR을 열라고 README가 안내한다. 즉 ggml 저장소는 llama.cpp의 하류 미러에 가깝다.

### vcpkg `ggml` 포트 (원문 `ports/ggml/vcpkg.json`, 조회 시점 버전 0.24.0)
```json
{
  "name": "ggml",
  "version": "0.24.0",
  "supports": "!uwp",
  "dependencies": [ {"name": "vcpkg-cmake", "host": true}, {"name": "vcpkg-cmake-config", "host": true} ],
  "features": {
    "blas":   { "dependencies": ["blas", "cblas"] },
    "cuda":   { "supports": "!(windows & staticcrt)", "dependencies": ["cuda"] },
    "metal":  { "supports": "osx" },
    "opencl": { "supports": "!arm32", "dependencies": ["opencl"] },
    "openmp": { "supports": "!osx" },
    "vulkan": { "dependencies": [ {"name":"ggml","host":true,"default-features":false,"features":["vulkan"]},
                                   {"name":"glslang","host":true}, {"name":"shaderc","host":true},
                                   "spirv-headers", "vulkan" ] }
  }
}
```
주목할 점:
- `vulkan` 기능이 **자기 자신을 host 의존성**으로 요구한다. 셰이더 생성 도구(`vulkan-shaders-gen`)를 호스트에서 빌드해 크로스 컴파일에 쓰기 위해서다. 우리도 코드 생성 도구가 있다면 같은 패턴이 필요하다.
- portfile은 `GGML_NATIVE`와 AVX/SSE/BMI2/FMA/F16C를 끈다. 배포용 바이너리는 실행 머신을 모르기 때문이다. ggml 자체는 이를 위해 `GGML_CPU_ALL_VARIANTS` + `GGML_BACKEND_DL`(런타임 ISA 선택)을 제공한다(`CMakeLists.txt:86,183`).
- 설치 후 헤더를 수정해 동적 링크를 맞춘다는 점은 상류 CMake가 "설치 후 소비"에 완전하지 않았음을 뜻한다.

### ggml-easy
- 헤더 온리 C++ 래퍼. `ggml_easy::ctx ctx(params); ctx.load_safetensors("model.safetensors");` 수준의 API.
- ggml을 submodule로 고정(특정 커밋)하므로 패키지 매니저 친화적이지 않다.
- GPU 기본 활성, 미지원 연산은 CPU 폴백. safetensors(F32/F16/BF16)를 GGUF 변환 없이 로딩.
- 우리에게 주는 교훈: "사용성 계층"은 별도 헤더/모듈로 분리하되, 코어와 같은 저장소에서 같은 버전으로 관리하는 편이 낫다.

### gemma.cpp
- 핵심 약 2K LoC, Gemma 계열 전용. 범용 텐서 라이브러리가 아니다.
- Highway로 SIMD를 한 번만 작성하고 런타임 디스패치. ggml의 `src/ggml-cpu/arch/{x86,arm,riscv,...}` 손코딩 커널 36,800줄과 대비된다.
- vcpkg에 `highway` 포트가 있으므로, C++23 프로젝트에서 SIMD 계층을 Highway에 위임하는 것은 현실적인 선택이다.

### MLX / candle
- 둘 다 "그래프 → 실행" 모델이지만 ggml보다 높은 수준의 배열 API를 제공한다.
- candle은 GGUF 양자화 파일을 읽고 CPU SIMD/CUDA/Metal 커널로 실행한다. **파일 포맷 호환은 유지하면서 구현 언어를 바꾸는** 것이 가능함을 증명한 사례다.

## 0.4 우리 프로젝트의 포지셔닝 제안

1. **호환 대상은 API가 아니라 포맷과 수치다.** GGUF 파일과 양자화 블록 레이아웃(`src/ggml-common.h`)은 그대로 읽고, 결과는 ggml과 비트 단위 또는 허용 오차 내로 일치시킨다. ggml의 C API를 흉내 낼 필요는 없다.
2. **ggml을 개발 의존성으로 vcpkg에서 가져와 차등 테스트(differential testing) 오라클로 쓴다.** `vcpkg.json`의 `"dependencies"`가 아닌 test feature에만 `ggml`을 넣는다.
3. **C++23 표준만으로 안 되는 것 세 가지**를 미리 결정한다: SIMD(`std::simd`는 C++26), 반정밀도(`std::float16_t`는 컴파일러별 지원 차이), 플러그인 ABI(백엔드 동적 로딩은 여전히 `extern "C"` 심볼이 필요).
4. 상류가 고정한 C++17 정책 때문에 우리 코드는 상류에 되돌릴 수 없다. 대신 **test-backend-ops의 테스트 케이스 정의**(`tests/test-backend-ops.cpp`)를 이식해 상류의 검증 자산을 재사용한다.

## 출처
- vcpkg PR: https://github.com/microsoft/vcpkg/pull/43925
- vcpkg ggml 포트 원문: https://github.com/microsoft/vcpkg/tree/master/ports/ggml
- ggml-easy: https://github.com/ngxson/ggml-easy , 소개 글 https://blog.ngxson.com/introducing-ggml-easy
- 연산자 오버로딩 RFC: https://github.com/ggml-org/ggml/issues/1326
- gemma.cpp: https://github.com/google/gemma.cpp
- MLX: https://github.com/ml-explore/mlx
- candle: https://github.com/huggingface/candle
- ZML: https://zml.ai/
- llama2.zig: https://github.com/cgbur/llama2.zig
- C++23 컴파일러 지원 현황: https://en.cppreference.com/cpp/compiler_support/23
