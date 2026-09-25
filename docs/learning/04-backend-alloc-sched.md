# 4. 백엔드 추상화, 할당기, 스케줄러

## 4.1 다섯 개의 인터페이스

`src/ggml-backend-impl.h`는 백엔드 구현체가 채워야 하는 함수 포인터 테이블 다섯 개를 정의한다. 모두 "인터페이스 구조체 + `void * context`" 패턴이다.

| 구조체 | 역할 | 필수 함수 |
|---|---|---|
| `ggml_backend_buffer_type_i` (`:17`) | 메모리 종류. "CUDA0 디바이스 메모리", "CPU 호스트 메모리", "CUDA 핀드 호스트 메모리" | `get_name`, `alloc_buffer`, `get_alignment` |
| `ggml_backend_buffer_i` (`:46`) | 할당된 메모리 블록. 텐서 읽기/쓰기 | `get_base`, `set_tensor`, `get_tensor`, `clear` |
| `ggml_backend_i` (`:121`) | 실행 스트림. 그래프 계산, 비동기 복사, 이벤트 | `get_name`, `free`, `graph_compute` |
| `ggml_backend_device_i` (`:176`) | 물리 디바이스. 능력 질의, 백엔드 생성 | `get_name`, `get_type`, `init_backend`, `get_buffer_type`, `supports_op`, `supports_buft` |
| `ggml_backend_reg_i` (`:230`) | 레지스트리(플러그인 단위). 디바이스 열거 | `get_name`, `get_device_count`, `get_device` |

관계: `reg → device[] → (backend, buffer_type[]) → buffer[]`.

`ggml_backend_dev_props`(`include/ggml-backend.h:162`)는 디바이스 타입(CPU/GPU/IGPU/ACCEL/META), 메모리, PCI id, 그리고 `caps`(async, host_buffer, buffer_from_host_ptr, events, mmap_support)를 알려 준다. 스케줄러가 이 정보로 복사 전략을 정한다.

`supports_op(dev, tensor)`가 특히 중요하다. 백엔드는 **연산별·타입별로 지원 여부를 동적으로 답**하고, 스케줄러가 미지원 연산을 CPU로 보낸다. 새 백엔드를 추가할 때 모든 연산을 구현하지 않아도 되는 이유다.

C++23 대응: 함수 포인터 테이블은 사실상 수동 vtable이다. 순수 가상 클래스로 바꾸면 자연스럽지만, **플러그인 경계**(4.2)는 C ABI여야 하므로 "내부는 `virtual`/concept, 경계는 `extern "C"` 어댑터" 구조가 된다.

## 4.2 레지스트리와 동적 로딩

- 정적 링크 시 `ggml-backend-reg.cpp:171`의 `#ifdef GGML_USE_CPU register_backend(ggml_backend_cpu_reg());`처럼 컴파일 타임에 등록된다.
- `GGML_BACKEND_DL=ON`이면 각 백엔드는 `MODULE` 라이브러리(`ggml-cuda.so`, `ggml-cpu-haswell.so` ...)가 되고, `ggml_backend_load_all()`(`:585-594`)이 이름 목록(blas, zendnn, cann, cuda, hip, metal, rpc, sycl, vulkan, virtgpu ...)을 돌며 `ggml_backend_load_best()`를 호출한다.
- `load_best`는 같은 이름의 후보 여러 개(예: CPU 변종 `ggml-cpu-sandybridge`, `-haswell`, `-icelake`)에서 `ggml_backend_score()` C 심볼(`:229`)이 가장 높은 것을 고른다. 점수 0은 "이 머신에서 못 씀". 그리고 `ggml_backend_init()` C 심볼(`:237`)로 `ggml_backend_reg_t`를 얻는다.
- 이 두 심볼은 `GGML_BACKEND_DL_IMPL(reg_fn)` 매크로(`ggml-backend-impl.h`)가 `extern "C"`로 만들어 준다. `GGML_BACKEND_API_VERSION = 2`로 ABI 버전을 확인한다.

vcpkg 관점: 포트가 네이티브 최적화를 끄는 이유가 여기서 이해된다. 범용 바이너리 + `GGML_CPU_ALL_VARIANTS` + 런타임 score 선택이 배포 가능한 조합이다.

## 4.3 텐서 할당기: `ggml_tallocr`와 `ggml_backend_alloc_ctx_tensors`

가장 단순한 경로: 컨텍스트에 `no_alloc = true`로 텐서를 만들고, `ggml_backend_alloc_ctx_tensors(ctx, backend)`(`include/ggml-alloc.h:82`)를 부르면 **컨텍스트의 모든 텐서를 하나의 백엔드 버퍼에 순서대로** 배치한다. 모델 가중치 로딩에 쓰는 패턴이다(`examples/gpt-2/main-backend.cpp`). 내부는 `ggml_tallocr`(선형 bump allocator, `ggml-alloc.h:14`)이다.

주의: 버퍼 타입의 `get_max_size`(예: Metal의 버퍼 상한)가 있으면 여러 버퍼로 쪼개져 `multi_buffer`(`ggml-backend-impl.h:88`)가 된다.

## 4.4 그래프 할당기: `ggml_gallocr`

계산 그래프의 **중간 텐서**는 수명이 짧다. `ggml_gallocr`(`src/ggml-alloc.c:482`)은 그래프를 한 번 훑어 각 텐서의 수명(마지막 사용 노드)을 계산하고, free-list 할당기(`ggml_dyn_tallocr`, `:121`, 청크당 최대 256개 free 블록)로 **주소를 재사용**한다. 실제 메모리는 `ggml_gallocr_reserve` 시점에 딱 한 번 잡힌다.

핵심 규칙 (`ggml_gallocr_allocate_node`, `:623`):
1. 뷰는 할당하지 않는다.
2. `ggml_op_can_inplace(op)`(`:22`)인 연산(ADD, MUL, SCALE, SOFT_MAX, RMS_NORM, ROPE, UNARY ...)은 **부모의 메모리를 그대로 출력으로 재사용**할 수 있다. 조건: 부모가 이 할당기 소유이고, OUTPUT 플래그가 없고, 레이아웃이 같고, 자식이 하나뿐(`n_children == 1 && n_views == 0`).
3. INPUT 플래그 텐서는 그래프 시작에 겹치지 않게, OUTPUT 플래그 텐서는 절대 해제하지 않는다.
4. 그래프 토폴로지가 바뀌면 자동 재할당(단일 버퍼일 때). `GGML_SCHED_NO_REALLOC` 옵션으로 디버깅 시 금지할 수 있다.

`ggml_op_can_inplace`가 참인 연산의 커널은 `restrict` 포인터를 쓰면 안 된다는 주석(`:21`)이 있다. 입출력이 같은 메모리일 수 있기 때문이다. C++23으로 옮길 때 `std::span`을 쓰면 aliasing 가정이 없으니 오히려 안전하다.

## 4.5 스케줄러: `ggml_backend_sched`

여러 백엔드(예: `{CUDA0, CUDA1, CPU}`)에 그래프를 나눠 실행한다. `simple-backend.cpp:60-63`이 최소 사용 예다.

```cpp
ggml_backend_t backends[2] = { model.backend, model.cpu_backend };
model.sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, /*parallel*/false, /*op_offload*/true);
...
ggml_backend_sched_reset(sched);
ggml_backend_sched_alloc_graph(sched, gf);
ggml_backend_tensor_set(model.a, matrix_A, 0, ggml_nbytes(model.a));  // 할당 후에 입력 복사
ggml_backend_sched_graph_compute(sched, gf);
```

분할 알고리즘 `ggml_backend_sched_split_graph`(`src/ggml-backend.cpp:1087-1297`)은 5 pass다.
1. **pass 1**: 이미 할당된 입력(가중치)이 있는 노드는 그 버퍼의 백엔드로. `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`로 표시된 버퍼가 우선.
2. **pass 2**: 이웃 노드로 배정 확장. GPU(우선순위 높은 백엔드)를 위아래로 퍼뜨리고 CPU는 무시.
3. **pass 3**: 호환 버퍼 타입이면 더 높은 우선순위 백엔드로 승격.
4. **pass 4**: 남은 `src`/`view_src`는 `dst`를 따라감.
5. **pass 5**: 백엔드가 바뀌는 지점마다 **split**을 만들고, 다른 백엔드에 있는 입력은 복사 텐서를 삽입(`GGML_SCHED_MAX_SPLIT_INPUTS = 30`).

실행 시 split마다 입력 복사(`ggml_backend_tensor_copy_async`) → `graph_compute` → 필요하면 이벤트로 동기화. `parallel=true`이면 여러 복사본(`n_copies`, `GGML_SCHED_MAX_COPIES`)으로 파이프라인 병렬을 한다.

`ggml_backend_sched_reserve(sched, worst_case_graph)`로 최대 크기 그래프를 미리 넣어 두면 실행 중 재할당이 없다. llama.cpp가 `n_ctx`, `n_batch` 최대치로 이렇게 한다.

디버깅: `GGML_SCHED_DEBUG=1` 환경변수로 split 결과를 출력하고, `ggml_backend_sched_set_eval_callback`으로 노드별 결과를 관찰할 수 있다(`include/ggml-backend.h:316`).

## 4.6 CPU 버퍼가 "항상 있는" 이유

`ggml_backend_cpu_buffer_type()`(`include/ggml-backend.h:433`)은 `ggml-base`에 포함돼 있어 CPU 백엔드 없이도 쓸 수 있다. 스케줄러의 마지막 폴백, GGUF mmap(`buffer_from_host_ptr`), 그리고 테스트에 필요하기 때문이다. 새 프로젝트도 "호스트 버퍼 타입"을 코어에 두어야 한다.

## 4.7 상호 검증 도구

`tests/test-backend-ops.cpp`는 **같은 그래프를 두 백엔드에서 실행하고 결과를 비교**한다(`ggml_backend_compare_graph_backend`, `include/ggml-backend.h:425`). 섹션 2에 연산별 테스트 케이스(`test_case` 상속)가 정의돼 있고, 섹션 3에서 shape/type 조합을 인스턴스화한다. 새 프로젝트는 이 케이스 목록을 이식해 "ggml CPU vs 우리 구현" 비교 테스트를 만들 수 있다.

## 이 장의 체크포인트

- [ ] `reg → device → backend/buffer_type → buffer` 관계를 그릴 수 있다.
- [ ] `supports_op`이 스케줄러의 CPU 폴백을 어떻게 가능하게 하는지 설명할 수 있다.
- [ ] `ggml_op_can_inplace`의 네 가지 재사용 조건을 말할 수 있다.
- [ ] 스케줄러 5 pass의 목적을 각각 한 문장으로 요약할 수 있다.
