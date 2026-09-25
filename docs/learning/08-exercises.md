# 8. 실습 과제

모든 실습은 이 저장소를 직접 빌드해 수행한다. 정답 출력은 실제로 실행해 확인한 값이다
(리눅스 x86-64, GCC, `GGML_NATIVE=OFF`, ggml v0.25.3).

## 준비: 빌드

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_BUILD_TESTS=ON -DGGML_BUILD_EXAMPLES=ON
cmake --build build -j8 --target simple-ctx simple-backend test-backend-ops test-quantize-fns
```

빌드 산출물:
- `build/bin/simple-ctx`, `build/bin/simple-backend`, `build/bin/test-backend-ops`, `build/bin/test-quantize-fns`
- `build/src/libggml.so`, `libggml-base.so`, `build/src/ggml-cpu/libggml-cpu.so`

자체 실습 프로그램은 다음처럼 컴파일한다(설치 없이 빌드 트리에 링크):
```bash
g++ -std=c++17 -O1 -Iinclude docs/learning/exercises/ex-strides-graph.cpp -o ex \
    -Lbuild/src -Lbuild/src/ggml-cpu -lggml -lggml-cpu -lggml-base \
    -Wl,-rpath,$PWD/build/src -Wl,-rpath,$PWD/build/src/ggml-cpu
./ex
```

---

## 실습 1. 예제 실행과 결과 해석 (1장, 3장)

`build/bin/simple-ctx`와 `build/bin/simple-backend`를 실행한다.

**정답 출력** (둘 다 동일):
```
mul mat (4 x 3) (transposed result):
[ 60.00 55.00 50.00 110.00
 90.00 54.00 54.00 126.00
 42.00 29.00 28.00 64.00 ]
```

질문:
1. `A`는 `ne=[2,4]`, `B`는 `ne=[2,3]`인데 결과가 "4 x 3"으로 출력되는 이유는? → 결과 `ne=[4,3]`이고 예제가 `ne[1]`을 행으로 찍기 때문. `ggml_mul_mat(A,B) = B·Aᵀ` 규약(3.1).
2. `simple-ctx`와 `simple-backend`의 차이를 함수 이름으로 말하라. → 전자는 `ggml_graph_compute_with_ctx`(CPU 전용, 데이터가 컨텍스트 안), 후자는 `ggml_backend_sched_alloc_graph` + `ggml_backend_tensor_set` + `ggml_backend_sched_graph_compute`(데이터가 백엔드 버퍼).
3. `simple-backend`에서 `ggml_backend_tensor_set`을 `ggml_backend_sched_alloc_graph` **앞**으로 옮기면 어떻게 되는가? → `data == NULL`이라 assert. 할당이 먼저다.

## 실습 2. 스트라이드와 뷰 (2장)

`docs/learning/exercises/ex-strides-graph.cpp`의 앞부분. F32 `[2,4]` 텐서, Q4_0 `[64]` 텐서, 전치, `cont`의 `ne`/`nb`를 출력한다.

**정답 출력**:
```
a        type=f32  ne=[2,4,1,1]  nb=[4,8,32,32]   contiguous=1 op=NONE
q4_0     type=q4_0 ne=[64,1,1,1] nb=[18,36,36,36] contiguous=1 op=NONE
a^T      type=f32  ne=[4,2,1,1]  nb=[8,4,32,32]   contiguous=0 op=TRANSPOSE
cont     type=f32  ne=[4,2,1,1]  nb=[4,16,32,32]  contiguous=1 op=CONT
row_size(Q4_0, 64) = 36, used_mem = 1584
```

확인할 것:
- `q4_0`의 `nb[0] = 18`: 원소 하나가 아니라 **블록 하나**(32원소)의 바이트다. `nb[1] = 64/32 * 18 = 36`.
- 전치 후 `nb[0] = 8 > nb[1] = 4`. 데이터는 그대로이고 스트라이드만 바뀌었다. `contiguous=0`.
- `cont`는 실제 복사 노드(`op=CONT`)를 만들고 `nb`를 정규화한다.
- `used_mem = 1584`: 텐서 헤더 4개(각 `ggml_tensor_overhead()` = `sizeof(ggml_object)` + `GGML_TENSOR_SIZE`) + 데이터(a 32B, q 36B, cont 32B) + 정렬 패딩. 직접 계산해 맞춰 보라.

추가 과제: `ggml_permute(ctx, t, 2, 0, 1, 3)`를 3차원 텐서에 적용해 `nb` 순열을 예측하고 검증하라.

## 실습 3. 작은 그래프 만들고 실행하기 (3장)

같은 파일의 뒷부분. `y = softmax(x·Wᵀ + b)`를 만든다. `x=[3,2]`, `W=[3,4]`, `b=[4]`.

**정답 출력**:
```
=== GRAPH ===
n_nodes = 3
 -   0: [     4,     2,     1]          MUL_MAT
 -   1: [     4,     2,     1]              ADD
 -   2: [     4,     2,     1]         SOFT_MAX
n_leafs = 3
 -   0: [     3,     4]     NONE           leaf_0
 -   1: [     3,     2]     NONE           leaf_1
 -   2: [     4,     1]     NONE           leaf_2
========================================
y        type=f32 ne=[4,2,1,1] nb=[4,16,32,32] contiguous=1 op=SOFT_MAX
y[0] = 0.2463 0.2487 0.2512 0.2538
y[1] = 0.2463 0.2487 0.2512 0.2538
```

확인할 것:
- `nodes[]`가 `MUL_MAT → ADD → SOFT_MAX` 후위 순서다. 리프(`op=NONE`)는 `leafs[]`에 따로 있다.
- `b`는 `[4]`인데 `[4,2]`에 더해졌다. `ggml_add`는 `src1`을 `src0`에 브로드캐스트한다(`ggml_can_repeat`).
- 두 행이 같은 이유: `W`의 각 행이 `[1,-0.5,-0.5]` 패턴이라 `x`의 두 배치가 같은 상대 로짓을 만든다. 데이터를 바꿔 달라지는지 확인하라.
- `graph.dot`이 생성된다. `dot -Tpng graph.dot -o graph.png`로 그리면 노드(흰색)와 리프(분홍)가 구분된다.

추가 과제: `ggml_set_output(y)`를 빼고 `ggml_gallocr`로 할당했을 때 결과가 깨지는지 실험하라(2.6절).

## 실습 4. 스레드 수와 배리어 (3장)

`ggml_graph_compute_with_ctx(ctx, gf, n_threads)`의 `n_threads`를 1, 2, 8로 바꿔 큰 `mul_mat`(예: `[4096,4096] × [4096,64]`)의 시간을 `ggml_time_us()`로 잰다.

확인할 것:
- 스케일링이 선형이 아닌 이유: 노드 사이 배리어, 메모리 대역폭, `mul_mat`의 청크 크기.
- `GGML_USE_OPENMP` 빌드(`-DGGML_OPENMP=ON`)와 자체 스레드풀 빌드의 차이.

## 실습 5. 백엔드 비교 테스트 (4장)

```bash
./build/bin/test-backend-ops -o MUL_MAT          # 모든 백엔드 쌍에 대해 MUL_MAT 비교
./build/bin/test-backend-ops -b CPU perf -o MUL_MAT
```

CPU만 있는 환경에서는 비교 대상이 없어 "skipped"가 많다. 대신 코드를 읽는다:
- 섹션 2의 `struct test_mul_mat : public test_case`를 찾아 `build_graph`, `initialize_tensors`, `max_nmse_err`가 무엇을 정의하는지 정리하라.
- 섹션 3에서 `MUL_MAT`이 몇 가지 (type, shape, batch) 조합으로 인스턴스화되는지 세어 보라.
- 이 케이스 목록이 새 프로젝트의 **테스트 스펙**이 된다(7장 로드맵 3단계).

## 실습 6. 양자화 왕복과 GGUF 헤더 (5장)

```bash
./build/bin/test-quantize-fns
```
각 타입에 대해 "quantization error", "dequantization error", "dot product error"가 임계값 이하인지 출력한다. Q4_0의 값을 기록해 두라. 새 프로젝트의 합격선이다.

GGUF 헤더 파싱: `examples/gpt-2/download-ggml-model.sh`로 모델을 받거나, 아무 `.gguf` 파일을 구해 다음 순서로 손 파싱한다.
1. 처음 4바이트 `"GGUF"`, 다음 4바이트 버전(=3).
2. i64 `n_tensors`, i64 `n_kv`.
3. KV를 `n_kv`번 읽는다(문자열은 u64 길이 + 바이트). `general.alignment`가 있으면 기록.
4. 텐서 정보를 `n_tensors`번 읽는다. 첫 텐서의 `offset`을 기록.
5. 현재 파일 위치를 alignment(기본 32)로 올림한 것이 데이터 블롭 시작. 첫 텐서 데이터 = 블롭 시작 + offset.
6. `gguf_init_from_file` + `gguf_get_data_offset` + `gguf_get_tensor_offset(ctx, 0)`의 값과 비교.

Python으로 해도 좋다(`gguf-py`는 llama.cpp 저장소에 있다). 목표는 스펙을 손으로 한 번 따라가는 것이다.

## 실습 7. 새 프로젝트 0단계 (6장, 7장)

빈 저장소에서:
1. `vcpkg.json`(코어 의존성 0, `tests` feature에 `catch2`, `ggml`), `CMakePresets.json`, `CMakeLists.txt`(`cxx_std_23`)를 만든다.
2. `tensor_meta` 구조체 하나와 `ne → nb` 계산 함수를 C++23으로 작성한다(`std::array<int64_t,4>`, `consteval` 타입 테이블).
3. Catch2 테스트에서 vcpkg의 `ggml`을 링크해 `ggml_new_tensor_*`의 `nb`와 비교한다. 43개 타입 × 여러 `ne0`.
4. `vcpkg install --x-feature=tests`가 세 삼중항에서 성공하는 CI를 만든다.

이 단계가 끝나면 7장 로드맵 1단계가 완료된 것이다.

## 실습 8. 읽기 과제 (전 장)

다음 함수를 위에서 아래로 읽고 각 함수가 하는 일을 한 줄로 요약한 표를 만든다.
- `ggml_new_tensor_impl` (`src/ggml.c:1766`)
- `ggml_visit_parents` / `ggml_build_forward_impl` (`src/ggml.c`, `ggml_build_forward_expand` 부근)
- `ggml_graph_plan` (`src/ggml-cpu/ggml-cpu.c:2815`)
- `ggml_graph_compute_thread` (`src/ggml-cpu/ggml-cpu.c:3109`)
- `ggml_compute_forward_mul_mat` (`src/ggml-cpu/ggml-cpu.c`, `current_chunk` 사용 부분 `:1370-1440`)
- `ggml_gallocr_allocate_node` (`src/ggml-alloc.c:623`)
- `ggml_backend_sched_split_graph` (`src/ggml-backend.cpp:1087`)
- `ggml_backend_load_best` (`src/ggml-backend-reg.cpp:480`)
- `gguf_init_from_file_impl` (`src/gguf.cpp`)

이 표가 곧 7장 매핑표의 "현재 구현" 열을 스스로 채운 것이다.
