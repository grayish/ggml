# 7. C++23 재구현 청사진

> 이 장은 앞 장에서 배운 ggml의 각 개념을 C++23의 어떤 구성요소로 옮길지, 어떤 순서로 만들지, 무엇이 표준만으로는 안 되는지를 정리한다. 설계 결정이 아니라 **결정을 위한 후보와 근거**다.
> 점진 경로(ggml 포크의 의존성을 vcpkg로 옮기는 것)는 9장에 있고, 그 결과물인 feature/의존성 구성은 이 장의 로드맵 0단계에 그대로 들어간다. 표의 "C++23 후보" 중 SIMD·GEMM·초월함수 행은 9장에서 vcpkg 라이브러리(Highway, oneDNN, sleef)로 대체하는 안과 함께 읽어야 한다.

## 7.1 개념 매핑표

| ggml 개념 (위치) | 현재 구현 | C++23 후보 | 비고 |
|---|---|---|---|
| `ggml_context` 아레나 (`ggml.c:975`) | `malloc` 1회 + `ggml_object` 연결 리스트 | `std::pmr::monotonic_buffer_resource` + 객체별 `std::pmr::polymorphic_allocator` | `ggml_reset` = `release()`. "런타임 할당 0" 불변식 유지 |
| `ggml_tensor::ne/nb` (`ggml.h:690`) | `int64_t[4]`, `size_t[4]` 바이트 스트라이드 | 메타데이터는 `std::array<int64_t,4>` 유지; 접근은 `std::mdspan<T, dextents<int64_t,4>, layout_stride>` 뷰를 **파생** | 양자화 타입은 원소 접근 불가 → 블록 단위 mdspan |
| `enum ggml_type`, `enum ggml_op` | C enum, 파일 포맷 값 | `enum class type : int32_t` (값 동일), `std::to_underlying` | GGUF 호환을 위해 번호 재사용 금지 |
| `ggml_type_traits` 테이블 (`ggml.h:2976`) | 런타임 `static const` 배열 | `constexpr std::array` + `consteval` 검증 (`blck_size * bits == type_size*8` 등) | `static_assert`로 표 자체를 검증 |
| `op_params` 64바이트 (`ggml.h:700`) | `memcpy` | 연산별 `struct` (trivially copyable, `sizeof <= 64` static_assert) + `std::bit_cast` | `std::variant`는 크기·ABI가 불안정하므로 저장소는 바이트 배열 유지 권장 |
| `ggml_status` | C enum 반환 | `std::expected<T, error>` (`error`에 `std::source_location`) | `GGML_ASSERT`의 abort는 `contract` 없으므로 커스텀 매크로 유지 |
| 연산 빌더 110개 (`ggml.c`) | 자유 함수 `ggml_xxx(ctx, ...)` | 자유 함수 유지 + 얇은 `tensor_ref` 메서드/연산자 (`a + b`, `matmul(w, x)`) | RFC #1326의 수요. 빌더는 "노드 생성만" 하는 규약 유지 |
| `ggml_cgraph` (`ggml-impl.h:341`) | 포인터 배열 + 해시셋 | `std::vector<tensor*>`(pmr) + `std::unordered_set` 또는 open-addressing 유지 | 노드 순서 = 위상 정렬 후위 순서 불변식 |
| 백엔드 vtable 5종 (`ggml-backend-impl.h`) | 함수 포인터 구조체 + `void* context` | 내부: 순수 가상 인터페이스 또는 `concept backend`; 경계: `extern "C"` 어댑터 | 플러그인 ABI는 C 유지 (7.3) |
| 레지스트리/`dlopen` (`ggml-backend-reg.cpp`) | `dl_get_sym("ggml_backend_init")` | 동일 방식. 심볼 이름과 `api_version` 유지 시 **ggml 플러그인 재사용 가능성**까지 열림 | 야심적이지만 검토 가치 있음 |
| `ggml_gallocr` (`ggml-alloc.c:482`) | free-list + 생명주기 분석 | 알고리즘 그대로, `std::vector`/`std::span` | 테스트: 동일 그래프에 동일 오프셋 산출 |
| 스케줄러 5-pass (`ggml-backend.cpp:1087`) | C++ 이미 | 알고리즘 그대로 이식 | `GGML_SCHED_DEBUG` 출력 형식을 맞추면 비교 테스트 가능 |
| 스레드풀/배리어 (`ggml-cpu.c:576, 3109`) | pthread + atomics + 스핀 | `std::jthread`, `std::stop_token`, `std::barrier`, `std::atomic<int>::wait/notify` | 스핀 폴링 정책은 벤치마크 후 결정 |
| 연산 커널 디스패치 (`ggml-cpu.c:1744`, `ops.cpp`) | `switch(op)` → `switch(type)` → 템플릿 | `template <op O> void compute(...)` 특수화 + `constexpr` 타입 디스패치 테이블 | 조합 폭발은 이미 템플릿화된 `ops.cpp`가 참고 |
| SIMD 커널 (`ggml-cpu/arch/*`) | 아키텍처별 손코딩 36,800줄 | **표준 없음** (`std::simd`는 C++26). 선택지: (a) 인트린식 손코딩 유지, (b) Highway(vcpkg), (c) `std::experimental::simd` | 0장 gemma.cpp 참고. 초기엔 스칼라 참조 + (b) |
| F16/BF16 | `uint16_t` + 변환 함수 | 저장은 `uint16_t` 유지, 연산은 `std::float16_t` 있을 때만 (`__STDCPP_FLOAT16_T__`) | MSVC 미지원 |
| 로깅 `ggml_log_set` | `printf` 스타일 콜백 | `std::format` + 콜백은 `std::string_view`로 | `std::print`는 콘솔 전용 |
| GGUF (`gguf.cpp`) | `FILE*` + 콜백 리더 | `std::span<const std::byte>` 파서, `std::expected`, mmap 어댑터 | 포맷 v3 바이트 호환 필수 |
| `ggml-cpp.h` 삭제자 | `unique_ptr` typedef 8개 | RAII 클래스 자체로 대체 | 새 프로젝트에서는 필요 없음 |
| 모듈 | 헤더 | `export module ggml23;` 고려 | 2026 기준 CMake 3.28+ 지원, `import std`는 MSVC 외 실험적. 초기엔 헤더 + 옵션 모듈 래퍼 |

## 7.2 단계별 로드맵

각 단계는 "ggml과 같은 결과"를 확인하는 테스트로 끝난다. ggml 자체는 vcpkg의 `ggml` 포트로 가져와 오라클로 쓴다.

| 단계 | 산출물 | 완료 기준 (테스트) | 참고 소스 |
|---|---|---|---|
| 0 | 저장소 뼈대: `vcpkg.json`, `CMakePresets.json`, `cxx_std_23`, Catch2, CI(3 삼중항), 오버레이 포트 | `vcpkg install ggml23 --overlay-ports=./ports` 성공, 소비자 예제 링크 | 6장 |
| 1 | 타입 시스템 + 텐서 메타데이터 + 아레나 | `ne`→`nb` 계산이 ggml과 동일 (모든 43 타입 × 여러 shape), `ggml_tensor_overhead()`와 동일 크기일 필요는 없음 | `ggml.c:1766-1842` |
| 2 | 뷰/permute/reshape/cont 빌더 + 그래프 구축 | 같은 빌더 호출 시 `nodes[]` 순서와 개수가 ggml과 동일 (DOT 덤프 비교) | `ggml.c:7337`, `ggml_graph_dump_dot` |
| 3 | 원소별/정규화/softmax/rope F32 참조 커널 + 단일 스레드 실행기 | `test-backend-ops`의 케이스를 이식해 오차 < 1e-6 | `ops.cpp`, `tests/test-backend-ops.cpp` 섹션 2 |
| 4 | `mul_mat` F32/F16 + 스레드풀(`std::jthread`/`std::barrier`) + `mul_mat` 동적 청크 | 결과 동일, 8스레드 스케일링 측정 | `ggml-cpu.c:1370-1440` |
| 5 | Q8_0, Q4_0 양자화/역양자화 + Q4_0×Q8_0 `vec_dot` (스칼라 → Highway) | `test-quantize-fns` 이식: RMSE, dot 오차 동일 수준. **바이트 단위 블록 호환** | `ggml-quants.c`, `ggml-common.h` |
| 6 | GGUF 리더(mmap) + `gguf` 메타 API | llama.cpp가 만든 실제 `.gguf`를 열어 텐서 이름/shape/offset이 `gguf_get_*`와 동일 | `gguf.cpp`, `docs/gguf.md` |
| 7 | 버퍼 타입/버퍼/디바이스/백엔드 인터페이스 + CPU 백엔드 + `gallocr` | 같은 그래프에서 텐서 오프셋 동일, 컴퓨트 버퍼 크기 동일 | `ggml-alloc.c`, `ggml-backend.cpp` |
| 8 | 스케줄러 + 두 번째 백엔드(예: Vulkan 또는 "느린 참조 백엔드") | split 개수/경계가 `GGML_SCHED_DEBUG` 출력과 동일 | `ggml-backend.cpp:1087-1297` |
| 9 | 예제: `mnist` 추론 → `gpt-2` 추론 (GGUF) | 상류 예제와 동일 토큰 출력 | `examples/mnist`, `examples/gpt-2` |
| 10 | 플러그인 ABI(`extern "C"` `ggml_backend_init`) + 포트 공개 | ggml의 CPU 플러그인(.so)을 우리 레지스트리로 로드 시도 | `ggml-backend-reg.cpp`, `GGML_BACKEND_DL_IMPL` |

단계 1~3까지가 **"ggml을 이해했다"의 증명**이고, 5까지 가면 llama.cpp 성능의 원천을 재현한 것이다.

## 7.3 표준 C++23만으로 안 되는 것 (미리 결정할 사항)

1. **SIMD**: `std::simd`는 C++26. 선택지는 인트린식/Highway/`std::experimental::simd`(GCC/Clang만). 권장: 참조 스칼라 커널을 먼저 만들고 Highway를 `highway` feature 뒤에 둔다. 코어 의존성 0개 원칙과 충돌하지 않는다.
2. **플러그인 ABI**: C++ 클래스는 컴파일러 간 ABI가 없다. 백엔드를 별도 `.so`로 로드하려면 경계는 `extern "C"` 함수 포인터 테이블이어야 한다. ggml의 `ggml_backend_i`가 바로 그것이므로, **경계 구조체는 ggml과 같은 형태를 유지**하고 내부에서만 C++을 쓴다.
3. **반정밀도**: `std::float16_t`는 선택 사항. 저장 타입은 `uint16_t`.
4. **`import std`**: MSVC만 안정적. 헤더로 시작하고 모듈은 옵션.
5. **mmap**: 표준에 없다. `<sys/mman.h>` / `MapViewOfFile` 어댑터 필요 (ggml도 llama.cpp 쪽에 둔다).
6. **정렬 할당**: `std::aligned_alloc`은 C++17부터 있지만 MSVC 미구현. `operator new(std::align_val_t)`를 쓴다.
7. **원자적 대기**: `std::atomic::wait/notify`는 C++20에 있지만 구현별 스핀/futex 정책이 다르다. ggml 배리어와 성능 비교 필요.

## 7.4 ggml에서 의도적으로 바꿀 만한 것

- `ggml_tensor`의 3중 역할(배열 뷰/그래프 노드/할당 단위) 분리 여부. 분리하면 `mdspan` 뷰가 자연스럽지만 할당기·스케줄러가 `view_src`/`src[]`를 한 구조에서 순회하는 코드를 다시 써야 한다. **1차 구현은 합쳐 두고, 뷰 타입만 파생**하는 것을 권장.
- `GGML_MAX_DIMS = 4`, `GGML_MAX_SRC = 10`, `GGML_MAX_NAME = 64` 같은 고정 상수는 유지. 헤더 크기 고정은 아레나 예약 계산의 전제다.
- `name[64]`는 `std::string_view`로 대체 불가(소유 필요). `std::array<char,64>` 유지 또는 인터닝.
- 오류 처리: `GGML_ASSERT` abort 대신 `std::expected`를 API 반환으로. 단 커널 내부의 불변식 위반은 여전히 즉시 중단이 맞다.
- 테스트: `test-backend-ops`의 "두 백엔드 비교" 프레임을 "ggml vs ggml23" 비교로 재활용.

## 7.5 vcpkg 포트로서의 자기 검증 목록

- [ ] 코어 의존성 0, feature별 의존성만 존재
- [ ] 정적/동적 모두 빌드, `BUILD_SHARED_LIBS` 존중
- [ ] `find_package(ggml23 CONFIG)` + `ggml23::core`, `ggml23::cpu` 타깃
- [ ] `write_basic_package_version_file(COMPATIBILITY SameMajorVersion)`
- [ ] 소비자 테스트 프로젝트를 CI에서 설치본으로 빌드
- [ ] 삼중항 3종(x64-linux, x64-windows, arm64-osx)에서 녹색
- [ ] `vcpkg format-manifest` 통과

## 이 장의 체크포인트

- [ ] 매핑표에서 "표준만으로 불가" 항목 7개를 나열할 수 있다.
- [ ] 로드맵 1~3단계의 완료 기준 테스트를 직접 작성할 수 있다.
- [ ] 플러그인 경계를 C ABI로 두는 이유를 컴파일러 ABI 관점에서 설명할 수 있다.
