# 1. 아키텍처 개관

## 1.1 세 개의 레이어

```
┌──────────────────────────────────────────────────────────────┐
│ 응용 (examples/, llama.cpp, whisper.cpp)                      │
│   모델 정의 = ggml 연산 빌더 호출 → ggml_cgraph                │
├──────────────────────────────────────────────────────────────┤
│ 백엔드 (src/ggml-cpu, ggml-cuda, ggml-metal, ggml-vulkan ...) │
│   ggml_backend_i / ggml_backend_device_i / ggml_backend_reg_i │
│   함수 포인터 테이블을 채워 등록. 각각 별도 CMake 타깃          │
├──────────────────────────────────────────────────────────────┤
│ ggml-base (src/ggml.c, ggml-alloc.c, ggml-backend.cpp,       │
│            ggml-quants.c, gguf.cpp, ggml-opt.cpp, threading)  │
│   백엔드 독립적인 코어. 텐서 메타데이터, 그래프, 할당기,        │
│   스케줄러, 양자화 참조 구현, 파일 포맷                         │
└──────────────────────────────────────────────────────────────┘
```

CMake 타깃도 이 구조를 따른다.
- `ggml-base`: 코어. `Threads::Threads`와 `m`만 링크한다(`src/CMakeLists.txt:612-616`).
- `ggml-cpu`, `ggml-cuda`, ...: `ggml_add_backend_library()`(`src/CMakeLists.txt:381`)로 만들어진다. `GGML_BACKEND_DL=ON`이면 `MODULE` 라이브러리(플러그인)로, 아니면 `ggml`에 정적/공유로 링크된다.
- `ggml`: 사용자가 링크하는 우산 타깃. 백엔드 레지스트리(`ggml-backend-reg.cpp`)를 포함한다.

## 1.2 "그래프를 만들고, 할당하고, 실행한다"

ggml의 모든 사용 흐름은 세 단계로 나뉜다. `examples/simple/simple-backend.cpp`가 정확히 이 순서다.

1. **정의**: `ggml_init()`으로 컨텍스트를 만들고, `ggml_new_tensor_*`와 `ggml_mul_mat()` 같은 빌더로 텐서 노드를 만든다. 이 단계에서는 **계산이 일어나지 않는다**. 텐서는 `op`와 `src[]`만 채워진 메타데이터다.
2. **할당**: `ggml_build_forward_expand()`로 그래프를 완성한 뒤, 그래프 할당기(`ggml_gallocr`) 또는 스케줄러(`ggml_backend_sched`)가 각 텐서의 `data` 포인터를 백엔드 버퍼 안의 오프셋으로 채운다.
3. **실행**: 백엔드가 `nodes[]`를 순서대로 순회하며 `op`에 맞는 커널을 호출한다.

핵심 설계 결정: **런타임 메모리 할당이 없다** (README의 "Zero memory allocations during runtime"). 정의 단계의 텐서 메타데이터는 컨텍스트가 소유한 고정 크기 아레나에, 데이터는 미리 예약한 백엔드 버퍼에 들어간다. 이는 C++23으로 옮길 때도 유지해야 할 첫 번째 불변식이다.

## 1.3 소스 파일별 책임

| 파일 | 언어 | 책임 |
|---|---|---|
| `src/ggml.c` | C | 컨텍스트/오브젝트 아레나, 텐서 생성, 110여 개 연산 빌더(`ggml_add`, `ggml_rope_ext` ...), 그래프 구축과 역전파(`ggml_build_backward_expand`), 그래프 출력/DOT 덤프 |
| `src/ggml-impl.h` | C/C++ 공용 | `ggml_cgraph` 구조체(`:341`), 해시셋, `ggml_set_op_params`(`:147`), 로깅, `GGML_ASSERT` |
| `src/ggml-alloc.c` | C | `ggml_tallocr`(선형), `ggml_dyn_tallocr`(free-list), `ggml_gallocr`(그래프 단위 생명주기 분석) |
| `src/ggml-backend-impl.h` | C/C++ 공용 | 백엔드 구현체가 채워야 하는 5개 인터페이스 구조체 |
| `src/ggml-backend.cpp` | C++ | 버퍼/백엔드 공용 함수, `ggml_backend_sched`(5-pass 분할 알고리즘 `:1087-1297`), 그래프 복사/비교 유틸 |
| `src/ggml-backend-reg.cpp` | C++ | 레지스트리, `ggml_backend_load_all()`이 `ggml-cuda.so` 같은 플러그인을 `dlopen`해 `ggml_backend_score`/`ggml_backend_init` 심볼로 등록 |
| `src/ggml-backend-meta.cpp` | C++ | 여러 디바이스를 하나로 묶는 텐서 병렬(meta) 백엔드 |
| `src/ggml-quants.c` | C | 각 양자화 타입의 `quantize_row_*`, `dequantize_row_*`, 참조 `vec_dot` |
| `src/ggml-common.h` | C | `block_q4_0` 등 블록 구조체와 `static_assert` 크기 검증(`:196-199`) |
| `src/gguf.cpp` | C++ | GGUF 읽기/쓰기, `gguf_init_from_file` |
| `src/ggml-opt.cpp` | C++ | 데이터셋, 손실, AdamW/SGD 옵티마이저 (학습용 고수준 API) |
| `src/ggml-threading.cpp` | C++ | `ggml_critical_section_start/end` (전역 뮤텍스) |
| `src/ggml-cpu/ggml-cpu.c` | C | 스레드풀(`ggml_threadpool`), `ggml_graph_plan`(`:2815`), `ggml_graph_compute`(`:3399`), 연산 디스패치 `ggml_compute_forward`(`:1744`), `mul_mat` |
| `src/ggml-cpu/ops.cpp` | C++ | 나머지 연산 커널. 템플릿으로 타입 조합 처리 |
| `src/ggml-cpu/binary-ops.cpp`, `unary-ops.cpp`, `vec.cpp` | C++ | 원소별 연산, 벡터 프리미티브 |
| `src/ggml-cpu/arch/<isa>/` | C/C++ | x86/arm/riscv/loongarch/powerpc/s390/wasm별 `quants.c`(vec_dot), `repack.cpp` |
| `src/ggml-cpu/repack.cpp` | C++ | Q4_0 가중치를 4x4/8x8 인터리브 레이아웃으로 런타임 재배치(`GGML_CPU_REPACK`) |

## 1.4 공개 헤더가 노출하는 것

- `ggml.h`: 타입/연산 enum, `ggml_tensor`, `ggml_init_params`, 텐서 생성/조회, 연산 빌더, 그래프 API, 양자화 API, 로깅, 스레드풀 파라미터.
- `ggml-alloc.h`: `ggml_tallocr`, `ggml_gallocr`, `ggml_backend_alloc_ctx_tensors` (컨텍스트의 모든 텐서를 버퍼 하나에 할당).
- `ggml-backend.h`: 버퍼 타입/버퍼/백엔드/이벤트/디바이스/레지스트리/스케줄러/meta 백엔드.
- `ggml-cpu.h`: `ggml_cplan`, `ggml_graph_plan`, `ggml_graph_compute`, 스레드풀, CPU 기능 질의(`ggml_cpu_has_avx2` 등).
- `gguf.h`: KV 메타데이터와 텐서 정보 읽기/쓰기.
- `ggml-opt.h`: 학습.
- `ggml-cpp.h`: `ggml_context_ptr` 등 8개 `unique_ptr` typedef. **상류가 제공하는 C++ 지원의 전부다.**

## 1.5 상류의 C → C++ 이동 현황

- 최상위 `src/`에서 C인 파일: `ggml.c`, `ggml-alloc.c`, `ggml-quants.c`. 나머지(`ggml.cpp`, `ggml-backend*.cpp`, `gguf.cpp`, `ggml-opt.cpp`, `ggml-threading.cpp`)는 C++.
- CPU 백엔드도 `ggml-cpu.c`(스레딩, mul_mat)만 C이고, 연산 커널은 `ops.cpp`(12,206줄)로 옮겨졌다. `binary-ops.cpp:25`처럼 `template <float (*op)(float,float), typename src0_t, ...>` 형태로 타입 조합을 템플릿화했다.
- 하지만 표준은 C++17에 고정(`src/CMakeLists.txt:609 "don't bump"`)이고, 공개 API는 `extern "C"`다. 이유는 (1) 바인딩(Python, Rust, Go)이 C ABI에 의존하고 (2) 플러그인 백엔드가 `ggml_backend_init` C 심볼로 로드되기 때문이다(`src/ggml-backend-impl.h`의 `GGML_BACKEND_DL_IMPL` 매크로).

## 1.6 이 장의 체크포인트

- [ ] `ggml-base`와 `ggml-cpu`가 별도 타깃인 이유를 설명할 수 있다.
- [ ] "정의 → 할당 → 실행" 세 단계에서 각각 어떤 함수가 호출되는지 `simple-backend.cpp`에서 짚을 수 있다.
- [ ] 상류가 C++17을 고정한 두 가지 이유를 말할 수 있다.
