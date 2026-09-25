# ggml 학습 가이드 — "pure C++23 + vcpkg" 재구현을 위한 준비

이 디렉터리는 ggml을 **vcpkg 생태계를 따르는 순수 C++23 프로젝트**로 다시 만들기 위해,
먼저 ggml 자체를 깊이 이해하는 것을 목표로 한 학습 자료입니다.
모든 문서는 이 저장소(ggml v0.25.3, `ggml : bump version to 0.25.3` 커밋 기준)의 실제 소스를 인용하며,
`파일:줄번호` 형식의 참조는 그대로 열어 볼 수 있습니다.

## 문서 목록

| # | 문서 | 한 줄 요약 | 예상 학습 시간 |
|---|------|-----------|--------------|
| 0 | [00-related-projects.md](00-related-projects.md) | 관련 프로젝트 탐색과 비교 (vcpkg 포트, C++ 래퍼, 타 언어 재구현, 경쟁 프레임워크) | 1시간 |
| 1 | [01-architecture.md](01-architecture.md) | 저장소 구조와 레이어(ggml-base / 백엔드 / 예제)를 한눈에 | 1시간 |
| 2 | [02-context-tensor.md](02-context-tensor.md) | `ggml_context` 아레나, `ggml_tensor`의 `ne`/`nb`, 뷰, 타입 시스템 | 2시간 |
| 3 | [03-graph-and-cpu-compute.md](03-graph-and-cpu-compute.md) | 연산 그래프 구축, `op_params`, CPU 백엔드의 스레드풀과 커널 디스패치 | 3시간 |
| 4 | [04-backend-alloc-sched.md](04-backend-alloc-sched.md) | 백엔드 추상화(vtable), 동적 로딩, 그래프 할당기, 스케줄러 | 3시간 |
| 5 | [05-quant-gguf.md](05-quant-gguf.md) | 양자화 블록 포맷, `ggml_type_traits`, GGUF 파일 포맷 | 2시간 |
| 6 | [06-build-vcpkg.md](06-build-vcpkg.md) | CMake 구조, 설치/패키지 설정, vcpkg 포트 분석과 새 프로젝트의 vcpkg 구성 | 2시간 |
| 7 | [07-cpp23-blueprint.md](07-cpp23-blueprint.md) | ggml 개념 → C++23 구성요소 매핑, 단계별 로드맵, 함정 목록 | 2시간 |
| 8 | [08-exercises.md](08-exercises.md) | 직접 빌드하고 실험하는 실습 과제 (정답 포함) | 4시간 이상 |

## 권장 학습 순서

1. **0 → 1**: 왜 이 프로젝트를 하는지, 이미 있는 것은 무엇인지 파악합니다.
2. **2 → 3**: `examples/simple/simple-ctx.cpp`를 옆에 두고 읽습니다. 이 두 장이 ggml의 절반입니다.
3. **4**: `examples/simple/simple-backend.cpp`와 `examples/gpt-2/main-sched.cpp`로 확장합니다.
4. **5**: `tests/test-quantize-fns.cpp`와 `docs/gguf.md`를 함께 읽습니다.
5. **6 → 7**: 새 프로젝트의 뼈대를 설계합니다.
6. **8**: 각 장의 실습을 순서대로 수행합니다. 실습 결과가 곧 새 프로젝트의 테스트 오라클이 됩니다.

## 핵심 파일 지도 (읽는 순서대로)

```
include/ggml.h              3022줄  공개 API: 타입, 텐서, 연산 빌더, 그래프
include/ggml-alloc.h          86줄  텐서/그래프 할당기
include/ggml-backend.h       437줄  백엔드/디바이스/레지스트리/스케줄러 API
include/ggml-cpu.h                  CPU 백엔드 전용 API (ggml_graph_plan / ggml_graph_compute)
include/gguf.h               211줄  GGUF 파일 포맷 API
include/ggml-cpp.h            39줄  C++용 unique_ptr 삭제자 (ggml이 제공하는 유일한 C++ 래퍼)
src/ggml-impl.h                     내부 구조체 (ggml_cgraph, hash set, op_params 접근자)
src/ggml.c                  8167줄  컨텍스트, 텐서 생성, 연산 빌더, 그래프 구축, 역전파
src/ggml-alloc.c            1249줄  동적 텐서 할당기 + 그래프 할당기
src/ggml-backend-impl.h             백엔드 구현체가 채워야 하는 함수 포인터 테이블
src/ggml-backend.cpp        2513줄  버퍼/백엔드 공용 로직 + 스케줄러
src/ggml-backend-reg.cpp            백엔드 레지스트리, 동적 라이브러리 로딩
src/ggml-quants.c           5638줄  양자화/역양자화 참조 구현
src/ggml-common.h                   양자화 블록 구조체 정의 (block_q4_0 등)
src/gguf.cpp                        GGUF 리더/라이터
src/ggml-cpu/ggml-cpu.c     3944줄  스레드풀, 그래프 계획, 연산 디스패치, mul_mat
src/ggml-cpu/ops.cpp       12206줄  나머지 연산 커널 (C++ 템플릿)
src/ggml-cpu/arch/*                 아키텍처별 SIMD 커널 (x86, arm, riscv, ...)
```
