# 9. ggml이 직접 만든 바퀴 목록과 vcpkg 대체재 지도

> 목표를 정확히 하면: **ggml을 vcpkg로 소비하는 것이 아니라, ggml(또는 그 후속) 자체가 의존성을 vcpkg에서 가져오고, 직접 구현한 부분 중 검증된 오픈소스로 대체 가능한 것을 대체해 "바퀴 재발명"을 줄이는 것**이다.
> 이 장은 (1) ggml이 손수 구현한 것의 목록, (2) 각각에 대한 vcpkg 포트 후보와 버전(2026-09 `microsoft/vcpkg` master 기준 직접 확인), (3) 성능 관점의 판정, (4) 교체 순서를 담는다.

## 9.1 한 장 요약

| 판정 | 항목 | 이유 |
|---|---|---|
| **즉시 교체 (성능 무관, 순수 이득)** | KleidiAI FetchContent → `kleidiai`, CCCL FetchContent → CUDA 툴킷 동봉본, stb → `stb`, CPU 기능 감지 → `cpuinfo`/`cpu-features`, 테스트 하네스 → `catch2`+`benchmark`, Vulkan 도구 체인 → `vulkan`/`shaderc`/`glslang`/`spirv-headers`, 로깅 → `fmt` | 빌드 재현성·유지보수. 실행 성능에 영향 없음 |
| **조건부 교체 (측정 후)** | F32/F16/BF16 대행렬 GEMM(llamafile sgemm) → `openblas`/`intel-mkl`/`onednn`, Vulkan 메모리 관리 → `vulkan-memory-allocator`, OpenCL F32 GEMM → `clblast`, 벡터 초월함수(expf/silu/gelu) → `sleef` 또는 Highway math, 해시셋 → `unordered-dense` | 상황에 따라 이득. 게이트(어떤 크기·타입에서 쓸지)를 ggml처럼 유지해야 함 |
| **전략적 결정 (파일럿 필요)** | 아키텍처별 SIMD 커널 36,800줄 → `highway`(또는 `xsimd`) | 코드 7분의 1로 줄고 런타임 디스패치를 공짜로 얻지만, 양자화 내적의 특수 명령(`vpdpbusd`, `sdot`, `i8mm`) 커버리지를 벤치마크로 확인해야 함 |
| **교체하지 않음 (ggml의 핵심 가치)** | 양자화 포맷과 Q4×Q8 내적, CUDA MMQ/FlashAttention, repack, 그래프 할당기, 스케줄러, GGUF, 노드 단위 배리어 스레드풀 | vcpkg에 동등한 것이 없거나(블록 양자화 커널), 있어도 더 느리다(범용 스레드풀) |

## 9.2 바퀴 목록 (전수)

크기는 `wc -l` 기준. "vcpkg" 열의 버전은 조회 시점의 포트 버전.

### A. 빌드/인프라 — 성능 무관, 즉시 교체 가능

| # | ggml 구현 | 위치 | 크기 | vcpkg 후보 | 판정 |
|---|---|---|---|---|---|
| A1 | KleidiAI를 `FetchContent`로 다운로드 (v1.24.0 고정) | `src/ggml-cpu/CMakeLists.txt:608-637` | - | `kleidiai` 1.25.0 | **교체**. `find_package(KleidiAI)`로 전환. 버전은 vcpkg baseline이 고정 |
| A2 | CCCL(cub/thrust)을 `FetchContent`로 (v3.2.0) | `src/ggml-cuda/CMakeLists.txt:63-72` | - | vcpkg 포트 없음. CUDA 툴킷에 동봉 | **교체**. vcpkg `cuda` 포트는 시스템 툴킷을 찾아 주는 메타 포트. 툴킷 동봉 CCCL을 쓰고 FetchContent 제거 |
| A3 | `stb_image.h`, `stb_image_write.h` 벤더링 | `examples/` | 2 파일 | `stb` 2024-07-29 | **교체** |
| A4 | CPU 기능 감지 (cpuid 파싱, `/proc/cpuinfo`, `getauxval`) | `src/ggml-cpu/arch/x86/cpu-feats.cpp`(327줄), `arm/cpu-feats.cpp`, `ggml-cpu.c` 산재 25곳 | ~400줄 | `cpuinfo`(PyTorch 계열, 2026-04), `cpu-features`(Google, 0.11.0) | **교체**. `ggml_backend_score()`와 `ggml_cpu_has_*()`의 구현체로. 캐시 토폴로지·코어 종류(P/E)까지 공짜로 얻음 |
| A5 | 자체 테스트 실행기, `add_test` 수동 등록 | `tests/CMakeLists.txt`, `tests/test-*.cpp` | - | `catch2` 3.16.0 또는 `gtest` 1.18.0, `benchmark` 1.9.5, `nanobench` | **교체**(점진). `test-backend-ops`의 케이스 정의는 유지하고 실행기만 Catch2로 |
| A6 | 로깅 `vsnprintf` 콜백 | `src/ggml.c:261,287-294` | ~80줄 | `fmt` 12.2.0 / `spdlog` 1.17.0 | **선택**. C++23이면 `std::format`으로 충분. C API 콜백 시그니처는 유지 |
| A7 | 동적 라이브러리 로딩 래퍼 | `src/ggml-backend-dl.cpp` | 48줄 | `dylib` 3.0.1, `boost-dll` | **유지**. 48줄을 위해 의존성을 늘일 이유 없음 |
| A8 | Vulkan 셰이더 도구체인 (`glslc`, SPIRV-Headers) | `src/ggml-vulkan/CMakeLists.txt:9,14` | - | `vulkan` 2023-12, `shaderc` 2026.2, `glslang` 16.4.0, `spirv-headers`, `volk` | **교체**. 이미 `find_package`라 vcpkg가 그대로 공급. `volk`로 로더 의존 제거 가능 |
| A9 | Windows용 LLVM OpenMP 설치기 다운로드 | `src/CMakeLists.txt:273,306` | - | 포트 없음 | **유지** 또는 MSVC OpenMP 사용 |
| A10 | ZenDNN, Hexagon HTP, virtgpu `ExternalProject` | 각 백엔드 CMake | - | 포트 없음 | **유지** (벤더 SDK) |
| A11 | 타이머(`QueryPerformanceCounter`/`clock_gettime`) | `src/ggml.c:528-541` | ~40줄 | `std::chrono` | **교체**(C++ 경로) |
| A12 | 테스트/예제의 `std::regex` 토크나이저 | `examples/common.cpp:137,248` | - | `re2` 2025-11, `sentencepiece` 0.2.1 | **선택**. 예제 품질 문제이지 코어 아님 |

### B. 연산 — vcpkg 대체재가 경쟁력 있는 경우 (게이트 필요)

| # | ggml 구현 | 위치 | 크기 | vcpkg 후보 | 판정과 근거 |
|---|---|---|---|---|---|
| B1 | F32/F16/BF16(+Q8_0/Q4_0/Q5_0/IQ4_NL) tinyBLAS GEMM (llamafile 기원, Apache-2 벤더링) | `src/ggml-cpu/llamafile/sgemm.cpp` | 4,164줄 | `openblas` 0.3.33, `intel-mkl` 2025.2, `onednn` 3.12, `eigen3` 5.0.1 | **조건부 교체**. ggml은 이미 `ggml-blas` 백엔드를 갖고 있고 `min_batch = 32`(`src/ggml-blas/ggml-blas.cpp:418`) 이상, `src1`이 F32일 때만 BLAS로 보낸다. 즉 **프롬프트 처리(대배치)는 BLAS가 이기고, 토큰 생성(행렬-벡터)은 메모리 대역폭이 지배해 ggml 커널이 이긴다**. llamafile sgemm은 이 중간 영역(작은 배치, 양자화 `src0`)을 채운다. 대체하려면 `onednn`(양자화 int8 GEMM, AMX 지원)이 가장 가깝고, `src/ggml-cpu/amx/`(AMX 전용 MMQ)도 oneDNN이 흡수 가능 |
| B2 | Vulkan 디바이스 메모리 관리 | `src/ggml-vulkan/ggml-vulkan.cpp` (16,299줄 중 버퍼/메모리 부분) | 수백 줄 | `vulkan-memory-allocator` 3.4.0 | **교체 권장**. VMA는 사실상 표준이고 서브할당·디프래그·메모리 타입 선택을 대신한다. ggml-vulkan은 `vk::Device::allocateMemory`를 직접 다룬다. 성능 저하 위험 낮음 |
| B3 | OpenCL 손코딩 커널 182개 | `src/ggml-opencl/kernels/` | - | `clblast` 1.7.0 | **부분 교체**. CLBlast는 F32/F16 GEMM만. 양자화 커널(대부분)은 대체 불가. Adreno 전용 튜닝이 많아 실익 작음 |
| B4 | 벡터 초월함수 `ggml_v_expf`/`silu`/`gelu` — SVE/NEON/AVX512/AVX2/SSE/RVV 6벌 + GELU F16 룩업 테이블 2개(각 128KB) | `src/ggml-cpu/vec.h:1097-1324, 62-65` | ~500줄 | `sleef` 3.9.0, `highway` contrib/math | **조건부 교체**. sleef는 ULP 보장과 벡터 길이별 심볼을 제공. 단 ggml 버전은 정밀도를 희생한 빠른 근사(`expf` 3~4 ULP)라 **정확도 요구가 다르다**. Highway `hwy/contrib/math`가 근사 수준이 비슷하고 런타임 디스패치를 포함해 더 적합 |
| B5 | CUDA 리덕션/스캔 | `src/ggml-cuda/{sum,mean,ssm-scan}.cu` | - | (툴킷 동봉 `cub`) | **이미 cub 사용**. A2로 공급 경로만 정리 |
| B6 | CUDA FlashAttention | `src/ggml-cuda/fattn-*.cu{,h}` | 수천 줄 | `cudnn` (vcpkg 포트는 7.6.5로 매우 오래됨 → 시스템 cuDNN 9 필요), `nvidia-cutlass` 4.8.0 | **유지**. ggml FA는 양자화 KV 캐시(Q8_0/Q4_0 K/V)를 직접 읽는다. cuDNN SDPA/CUTLASS FMHA는 F16/BF16/FP8만. 대체하면 KV 양자화를 잃음 |
| B7 | 포인터 해시셋 (open addressing, 비트셋) | `src/ggml-impl.h:238`, `ggml.c` 7곳 | ~150줄 | `unordered-dense` 5.0.1, `parallel-hashmap` 2.0.0, `abseil` | **선택**. 그래프 구축·할당 시에만 쓰이며 핫패스 아님. C 코어에 C++ 컨테이너를 넣으려면 경계 필요. C++23 경로에서만 |
| B8 | mmap (llama.cpp `llama-mmap.cpp`에 있음, ggml 코어엔 없음) | - | - | `mio` 2023-03 | **교체**(상위 프로젝트에서). 헤더 온리, Windows/POSIX 통합 |

### C. 전략적 결정 — 포터블 SIMD 계층

| # | ggml 구현 | 위치 | 크기 | vcpkg 후보 |
|---|---|---|---|---|
| C1 | 아키텍처별 손코딩 커널: x86 10,842 / arm 9,825 / riscv 8,337 / powerpc 2,386 / loongarch 2,309 / s390 1,809 / wasm 1,292 | `src/ggml-cpu/arch/*` | **36,800줄** | `highway` 1.4.0, `xsimd` 14.3.0 |
| C2 | SIMD 추상 매크로 (`GGML_F32x8_FMA` 등) | `src/ggml-cpu/simd-mappings.h`, `simd-gemm.h`, `vec.h` | 3,115줄 | 위와 동일 |

판정: **파일럿 후 결정**. 근거:
- Highway는 (a) 한 번 작성 → 모든 ISA, (b) 런타임 디스패치(`HWY_DYNAMIC_DISPATCH`) 내장으로 `GGML_CPU_ALL_VARIANTS` + `ggml_backend_score` 메커니즘(4.2절)을 대체, (c) gemma.cpp가 같은 방식으로 경쟁력 있는 성능을 냈다.
- 위험은 양자화 내적이다. `vec_dot_q4_0_q8_0`는 x86에서 `vpdpbusd`(VNNI), ARM에서 `sdot`/`i8mm`를 직접 쓴다. Highway는 `SatWidenMulPairwiseAdd`, `ReorderWidenMulAccumulate`, `SumOfMulQuadAccumulate` 등으로 이를 노출하지만 **모든 조합이 네이티브 명령으로 내려가는지는 타깃별로 다르다**.
- 따라서 파일럿: `vec_dot_q4_0_q8_0`와 `vec_dot_q8_0_q8_0` 두 커널만 Highway로 작성 → `tests/test-quantize-perf.cpp`와 `test-backend-ops perf -o MUL_MAT`로 x86(AVX2/AVX512-VNNI)과 ARM(NEON dotprod/i8mm)에서 기존 `arch/*` 대비 측정. 95% 이상이면 확대, 아니면 Highway는 원소별/정규화/softmax 커널(`ops.cpp`, `vec.h`)에만 적용하고 내적은 손코딩 유지.
- `xsimd`는 디스패치가 없고 정수 dot 계열 추상화가 약해 후순위.

### D. 교체하지 않을 것 — ggml의 핵심 가치

| 항목 | 이유 |
|---|---|
| 양자화 블록 포맷과 `quantize_row_*`/`dequantize_row_*` (`src/ggml-quants.c`, `ggml-common.h`) | GGUF 생태계 호환의 본체. 대체재 없음 |
| Q4×Q8 계열 정수 내적, K-quant, IQ 코드북 커널 | vcpkg의 어떤 BLAS/DNN도 이 블록 포맷을 모른다 |
| CUDA MMQ(`mmq-config-*.cuh`), MMVQ, FlashAttention | 양자화 입력을 직접 소비. CUTLASS/cuDNN은 F16/FP8 중심 |
| `repack.cpp` (Q4_0 → 4x4/8x8 인터리브) | 특정 커널 전용 레이아웃 |
| 그래프 할당기, 스케줄러, GGUF | 도메인 특화 알고리즘. 범용 라이브러리로 얻을 것 없음 |
| 스레드풀 + 노드 단위 배리어 (`src/ggml-cpu/ggml-cpu.c:481-505`) | `tbb` 2023.1/`taskflow` 4.1.0은 작업 훔치기(work-stealing) 스케줄러라 "모든 스레드가 같은 노드를 나눠 처리 → 배리어" 모델에 오버헤드가 크다. 선택지는 이미 있는 OpenMP(`GGML_OPENMP`) 또는 C++23의 `std::barrier`. 친화도·폴링·우선순위 제어(`prio`, `poll`)는 ggml 것을 유지 |
| F16 변환 (`src/ggml-impl.h:396-448`) | `half` 2.2.1은 소프트웨어 구현이라 F16C/NEON 하드웨어 변환보다 느림. Highway를 도입하면 벡터 변환은 거기서 얻음 |

## 9.3 포크용 `vcpkg.json` 초안

코어 의존성은 여전히 0이고, 모든 외부 라이브러리는 feature 뒤에 둔다. 이렇게 해야 "no dependencies" 빌드가 보존되고 vcpkg 포트 심사도 통과한다.

```json
{
  "name": "ggml-vx",
  "version-semver": "0.25.3",
  "description": "ggml with vcpkg-managed dependencies",
  "license": "MIT",
  "dependencies": [],
  "default-features": [],
  "features": {
    "cpuinfo":  { "description": "CPU feature/topology detection via cpuinfo", "dependencies": ["cpuinfo"] },
    "kleidiai": { "description": "Arm KleidiAI kernels", "supports": "arm64", "dependencies": ["kleidiai"] },
    "highway":  { "description": "Portable SIMD kernels via Highway (pilot)", "dependencies": ["highway"] },
    "sleef":    { "description": "Vectorized transcendental functions", "dependencies": ["sleef"] },
    "openblas": { "description": "BLAS backend (OpenBLAS)", "dependencies": ["openblas"] },
    "mkl":      { "description": "BLAS backend (Intel MKL)", "supports": "x64", "dependencies": ["intel-mkl"] },
    "onednn":   { "description": "int8/AMX GEMM via oneDNN", "supports": "x64", "dependencies": ["onednn"] },
    "cuda":     { "description": "CUDA backend (system toolkit)", "dependencies": ["cuda"] },
    "vulkan":   { "description": "Vulkan backend",
                  "dependencies": ["vulkan", "spirv-headers", "vulkan-memory-allocator", "volk",
                                   {"name": "shaderc", "host": true}, {"name": "glslang", "host": true},
                                   {"name": "ggml-vx", "host": true, "default-features": false, "features": ["vulkan-tools"]}] },
    "vulkan-tools": { "description": "Host-side vulkan-shaders-gen" },
    "opencl":   { "description": "OpenCL backend", "dependencies": ["opencl", "clblast"] },
    "examples": { "description": "Build examples", "dependencies": ["stb"] },
    "tests":    { "description": "Build tests and benchmarks", "dependencies": ["catch2", "benchmark"] }
  }
}
```

포트 → CMake 타깃 대응 (portfile/CMake에서 쓸 이름):

| 포트 | `find_package` | 타깃 |
|---|---|---|
| cpuinfo | `find_package(cpuinfo CONFIG)` | `cpuinfo::cpuinfo` |
| cpu-features | `find_package(CpuFeatures CONFIG)` | `CpuFeatures::cpu_features` |
| kleidiai | `find_package(KleidiAI CONFIG)` | `KleidiAI::kleidiai` (포트의 usage 파일 확인) |
| highway | `find_package(hwy CONFIG)` | `hwy::hwy`, `hwy::hwy_contrib` |
| sleef | `find_package(sleef CONFIG)` | `sleef::sleef` |
| openblas | `find_package(OpenBLAS CONFIG)` 또는 `find_package(BLAS)` | `OpenBLAS::OpenBLAS` |
| intel-mkl | `find_package(MKL CONFIG)` | `MKL::MKL` |
| onednn | `find_package(dnnl CONFIG)` | `DNNL::dnnl` |
| vulkan-memory-allocator | `find_package(VulkanMemoryAllocator CONFIG)` | `GPUOpen::VulkanMemoryAllocator` |
| volk | `find_package(volk CONFIG)` | `volk::volk_headers` |
| shaderc / glslang | `find_package(unofficial-shaderc CONFIG)`, `find_package(glslang CONFIG)` | `unofficial::shaderc::shaderc`, `glslang::glslang` |
| clblast | `find_package(CLBlast CONFIG)` | `clblast` |
| stb | `find_package(Stb)` | 헤더 경로 `Stb_INCLUDE_DIR` |
| catch2 / benchmark | `find_package(Catch2 3 CONFIG)`, `find_package(benchmark CONFIG)` | `Catch2::Catch2WithMain`, `benchmark::benchmark` |
| fmt | `find_package(fmt CONFIG)` | `fmt::fmt` |
| mio | `find_package(mio CONFIG)` | `mio::mio` |

기존 ggml의 `GGML_BLAS_VENDOR`/`pkg_check_modules` 경로(`src/ggml-blas/CMakeLists.txt:8-34`)는 vcpkg의 `find_package(BLAS)`가 OpenBLAS/MKL을 모두 처리하므로 단순화된다.

## 9.4 네이티브 최적화와 vcpkg의 충돌 해결

vcpkg는 바이너리 캐시를 위해 **빌드 환경과 무관한 산출물**을 전제한다. `GGML_NATIVE=ON`(`-march=native`)은 이와 충돌한다(vcpkg 포트가 이를 끄는 이유, 0장). 선택지:
1. **런타임 디스패치로 전환**: Highway(C1) 또는 기존 `GGML_CPU_ALL_VARIANTS` + `GGML_BACKEND_DL`. 배포 바이너리가 범용이면서 최고 ISA를 쓴다. **권장**.
2. 오버레이 삼중항: `triplets/x64-linux-native.cmake`에 `set(VCPKG_CXX_FLAGS "-march=native")`. 바이너리 캐시를 그 머신에만 유효하게 만든다. 개발자 로컬 용도.
3. `native` feature: 포트 옵션으로 노출. vcpkg 큐레이션 레지스트리에는 못 넣지만 자체 레지스트리에서는 가능.

## 9.5 교체 순서 (측정 기준 포함)

| 단계 | 작업 | 측정/합격 기준 | 위험 |
|---|---|---|---|
| 1 | A1, A2, A3, A8: 모든 `FetchContent`/`ExternalProject`(벤더 SDK 제외)와 벤더링 헤더를 vcpkg로 | `vcpkg install --x-feature=...` 3 삼중항 녹색, 빌드 결과 바이트 동일 여부 무관 | 낮음 |
| 2 | A4: `cpuinfo`로 `ggml_cpu_has_*`, `ggml_backend_score` 재구현 | `test-backend-ops`가 같은 백엔드 변종 선택 | 낮음 |
| 3 | A5: Catch2 + benchmark로 실행기 교체, `test-backend-ops` 케이스 유지 | 케이스 수 동일, CI 시간 | 낮음 |
| 4 | B2: Vulkan에 VMA 도입 | `test-backend-ops -b Vulkan0` 전 통과, `llama-bench` 토큰/초 ±2% | 중간 |
| 5 | B1: BLAS 백엔드를 vcpkg `openblas`/`mkl`/`onednn`으로, 게이트 `min_batch` 재튜닝 | 프롬프트 처리 t/s 향상, 토큰 생성 t/s 불변 | 중간 (게이트 잘못 잡으면 토큰 생성 퇴보) |
| 6 | C1 파일럿: `vec_dot_q4_0_q8_0`, `vec_dot_q8_0_q8_0`를 Highway로 | `test-quantize-perf` GB/s가 `arch/x86`, `arch/arm` 대비 ≥95% | 높음 (결과가 전략을 결정) |
| 7 | B4: 원소별/softmax/norm 커널을 Highway math 또는 sleef로 | `test-backend-ops` 오차 임계 통과, perf 모드 불변 | 중간 |
| 8 | C1 확대 또는 중단 | 6단계 결과에 따름 | - |
| 9 | 자체 vcpkg 레지스트리에 `ggml-vx` 포트 공개, 소비자 예제(`examples/test-cmake`) CI | 설치본으로 소비자 빌드 성공 | 낮음 |

측정 도구는 모두 저장소에 있다: `tests/test-backend-ops.cpp`(perf 모드), `tests/test-quantize-perf.cpp`, 그리고 상위 프로젝트의 `llama-bench`.

## 9.6 vcpkg 자체의 함정

- **`cuda` 포트는 툴킷을 설치하지 않는다.** 시스템에 설치된 CUDA를 찾아 주는 메타 포트다. CI 이미지에 툴킷이 있어야 한다.
- **`cudnn` 포트는 7.6.5로 낡았다.** cuDNN 9가 필요하면 시스템 설치 + `find_package(CUDNN)`을 직접 쓴다.
- **`intel-mkl`은 시스템 설치본을 찾는 포트**이며 라이선스(ISSL)를 확인해야 한다. 재배포 가능한 대안은 `openblas`.
- **정적/동적 CRT 삼중항 불일치**(`x64-windows` vs `x64-windows-static`)가 백엔드 플러그인(`GGML_BACKEND_DL`)과 충돌할 수 있다. 플러그인 모델을 쓰려면 동적 삼중항.
- **바이너리 캐시와 `-march=native`**: 9.4 참조.
- **`nvidia-cutlass`, `highway`는 헤더 중심이라 컴파일 시간이 크게 는다.** 미리 컴파일된 커널 라이브러리로 분리(예: `ggml-cpu-hwy` 타깃)해 증분 빌드를 지킨다.
- `kleidiai` 포트 버전(1.25.0)이 ggml이 고정한 v1.24.0보다 앞선다. API 차이가 있으면 `"version>="`로 하한만 걸고 코드를 맞춘다.

## 9.7 llama.cpp까지 확장할 때

상위 프로젝트가 벤더링한 것도 모두 vcpkg에 있다: `cpp-httplib` 0.58.0, `nlohmann-json` 3.12.0, `miniaudio` 0.11.25, `stb`. vcpkg의 `llama-cpp` 포트가 이미 `cpp-httplib`, `nlohmann-json`을 외부 의존성으로 바꾸는 패치를 들고 있으므로(0장), 그 패치가 곧 출발점이다.

## 이 장의 체크포인트

- [ ] "즉시 교체" 항목 7개를 나열하고 각각의 vcpkg 포트 이름을 말할 수 있다.
- [ ] BLAS가 토큰 생성에서는 ggml 커널을 이기지 못하는 이유(메모리 대역폭)와 ggml의 `min_batch` 게이트를 설명할 수 있다.
- [ ] Highway 파일럿의 합격 기준과, 실패 시 대안(원소별 커널에만 적용)을 말할 수 있다.
- [ ] `GGML_NATIVE`가 vcpkg 바이너리 캐시와 충돌하는 이유와 세 가지 해결책을 말할 수 있다.
