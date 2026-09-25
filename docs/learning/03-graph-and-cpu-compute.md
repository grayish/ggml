# 3. 연산 그래프와 CPU 실행

## 3.1 연산 빌더는 "노드를 만들 뿐"이다

`ggml_add(ctx, a, b)`를 호출하면 `src/ggml.c`의 빌더가:
1. 결과 shape를 계산해 `ggml_new_tensor_impl`로 헤더를 만든다 (`no_alloc`이면 데이터 없음).
2. `result->op = GGML_OP_ADD; result->src[0] = a; result->src[1] = b;`를 채운다.
3. 필요하면 `ggml_set_op_params(result, &params, sizeof(params))`(`src/ggml-impl.h:147`)로 64바이트 `op_params`에 인자를 `memcpy`한다. 예: `ggml_rope_ext`는 `n_dims, mode, n_ctx_orig, freq_base, ...`를 int32/float 배열로 밀어 넣고(`src/ggml.c:4315`), 커널은 `ggml_get_op_params_i32(a, 2)`로 꺼낸다.

`op_params`는 타입이 없는 바이트 배열이다. C++23에서는 `std::variant`나 연산별 `struct` + `std::bit_cast`로 대체할 수 있지만, 64바이트 고정 크기라는 제약은 **텐서 헤더 크기를 고정**하기 위한 것임을 기억하라.

### `ggml_mul_mat`의 규약 (가장 많이 틀리는 곳)

```c
// include/ggml.h:1479
// A: k columns, n rows => [ne03, ne02, n, k]
// B: k columns, m rows (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]
// result is n columns, m rows => [ne03 * x, ne02 * y, m, n]
```

즉 `ggml_mul_mat(A, B) = B · Aᵀ` 이고 두 입력 모두 `ne[0]`(안쪽 차원)이 같아야 한다. `simple-ctx.cpp`에서 `A`가 `[2,4]`, `B`가 `[2,3]`이면 결과는 `ne = [4, 3]`이다. 가중치 행렬을 `[in_features, out_features]`로 저장하면 `ggml_mul_mat(W, x)`가 `x · Wᵀ`, 곧 PyTorch `Linear`가 된다. 배치 차원(`ne[2]`, `ne[3]`)은 브로드캐스트된다(`x`, `y` 배수).

## 3.2 그래프 구축

```c
// src/ggml-impl.h:341
struct ggml_cgraph {
    int size, n_nodes, n_leafs;
    struct ggml_tensor ** nodes;      // 계산이 필요한 텐서 (op != NONE)
    struct ggml_tensor ** grads;      // 역전파용
    struct ggml_tensor ** grad_accs;
    struct ggml_tensor ** leafs;      // 상수/입력 (op == NONE)
    int32_t             * use_counts; // 해시 슬롯별 사용 횟수
    struct ggml_hash_set visited_hash_set;
    enum ggml_cgraph_eval_order order;
    uint64_t uid;
};
```

- `ggml_new_graph(ctx)`는 `GGML_DEFAULT_GRAPH_SIZE = 2048` 노드 분량을 컨텍스트 아레나에 잡는다. `ggml_new_graph_custom(ctx, size, grads)`로 조절한다.
- `ggml_build_forward_expand(gf, t)`(`src/ggml.c:7337`)는 `t`에서 출발해 `src[]`를 재귀적으로 방문(`ggml_visit_parents`)하며 **후위 순서(post-order)** 로 `nodes[]`에 넣는다. 따라서 `nodes[i]`의 모든 입력은 `nodes[<i]` 또는 `leafs[]`에 있다. 실행기는 이 배열을 앞에서 뒤로 순회만 하면 된다.
- 방문 여부는 포인터 해시셋으로 판별한다. 같은 텐서를 두 번 expand해도 중복되지 않는다.
- `ggml_graph_node(gf, -1)`은 마지막 노드, 곧 보통 최종 출력이다.

역전파: `ggml_build_backward_expand`가 각 노드의 `op`에 대한 미분 규칙을 적용해 `grads[]`를 채운다. 학습이 목표가 아니면 처음엔 건너뛰어도 된다.

## 3.3 CPU 백엔드: 계획(plan) → 실행(compute)

CPU 경로는 두 함수로 이뤄진다(`include/ggml-cpu.h`).

```c
struct ggml_cplan ggml_graph_plan(const struct ggml_cgraph *, int n_threads, struct ggml_threadpool *); // ggml-cpu.c:2815
enum ggml_status   ggml_graph_compute(struct ggml_cgraph *, struct ggml_cplan *);                       // ggml-cpu.c:3399
```

`ggml_graph_plan`은 노드를 순회하며
- 연산별 최대 스레드 수(`n_tasks`)를 정하고,
- 연산별 **작업 버퍼(work buffer) 크기**를 합산한다. 예: `mul_mat`은 `src1`을 `vec_dot_type`으로 양자화한 사본을 둘 공간이 필요하다.

`ggml_cplan`(`include/ggml-cpu.h:12`)에는 `work_size`, `work_data`, `n_threads`, `threadpool`, `abort_callback`, `use_ref`가 들어간다. `work_data`는 **호출자가** 확보한다. 백엔드 래퍼는 이렇게 한다:

```cpp
// src/ggml-cpu/ggml-cpu.cpp:170
struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads, cpu_ctx->threadpool);
if (cpu_ctx->work_size < cplan.work_size) { delete[] cpu_ctx->work_data; cpu_ctx->work_data = new uint8_t[cplan.work_size]; ... }
cplan.work_data = cpu_ctx->work_data;
return ggml_graph_compute(cgraph, &cplan);
```

`ggml_graph_compute_with_ctx(ctx, gf, n_threads)`는 작업 버퍼를 컨텍스트 아레나(`GGML_OBJECT_TYPE_WORK_BUFFER`)에서 잡는 편의 함수다. `simple-ctx.cpp:67`이 이를 쓴다.

## 3.4 스레드풀과 노드 단위 배리어

```c
// src/ggml-cpu/ggml-cpu.c:3109
static thread_ret_t ggml_graph_compute_thread(void * data) {
    ...
    struct ggml_compute_params params = { .ith = state->ith, .nth = n_threads, .wsize = cplan->work_size, .wdata = cplan->work_data, .threadpool = tp, .use_ref = cplan->use_ref };
    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load(&tp->abort) != node_n; node_n++) {
        struct ggml_tensor * node = cgraph->nodes[node_n];
        if (ggml_op_is_empty(node->op)) continue;           // VIEW/RESHAPE/PERMUTE 등은 계산 없음
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;
        ... ggml_compute_forward(&params, node); ...
        ggml_barrier(tp);                                   // 모든 스레드가 이 노드를 끝낼 때까지 대기
    }
}
```

실행 모델의 핵심:
- **모든 스레드가 모든 노드를 방문**한다. 노드 안에서 `ith/nth`로 행(row) 범위를 나눠 갖는다 (`ops.cpp:31-35`: `dr = (nk + nth - 1)/nth; k0 = dr*ith; k1 = MIN(k0+dr, nk)`).
- 노드 사이에는 `ggml_barrier`(`ggml-cpu.c:576`)가 있다. 스핀 + 원자 카운터 기반이다. 그래프 수준 병렬(독립 노드 동시 실행)은 하지 않는다.
- `mul_mat`만 예외적으로 **동적 청크 분배**를 한다. `threadpool->current_chunk`(`:492`) 원자 카운터를 `fetch_add`해 남은 청크를 가져간다. 스레드 간 부하 불균형을 줄이기 위해서다.
- `GGML_USE_OPENMP`가 켜져 있으면 `#pragma omp parallel`이 스레드 생성을 대신한다(`:3427`). 아니면 `pthread` 기반 자체 풀이 CPU 마스크·우선순위·폴링 정책을 가진다.

C++23 대응: `std::jthread` + `std::barrier` + `std::stop_token`이 이 구조를 거의 그대로 표현한다. 단 ggml의 배리어는 폴링(스핀) 시간을 `n_polling` 파라미터로 조정하므로, `std::barrier`의 블로킹 대기와 지연 특성이 다르다. 벤치마크 없이 바꾸면 안 된다.

## 3.5 연산 디스패치

```c
// src/ggml-cpu/ggml-cpu.c:1744
static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) return;
    if (ggml_cpu_extra_compute_forward(params, tensor)) return;   // repack/AMX/KleidiAI 등 "extra buffer type" 우선
    switch (tensor->op) {
        case GGML_OP_DUP: ggml_compute_forward_dup(params, tensor); break;
        case GGML_OP_ADD: ggml_compute_forward_add(params, tensor); break;
        ...
    }
}
```

각 `ggml_compute_forward_<op>`는 다시 **`src0->type`으로 switch**해 F32/F16/BF16/양자화별 구현으로 간다. `ops.cpp`는 이 조합 폭발을 템플릿으로 줄였다:

```cpp
// src/ggml-cpu/binary-ops.cpp:25
template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static void apply_binary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    ...
    if constexpr (std::is_same_v<src0_t, float> && ...) { /* 벡터화 경로 */ }
}
```

C++23에서는 이 부분이 가장 깔끔해진다. `enum class op`를 `template <op O>` 특수화와 `concept`으로 묶고, 타입 조합은 `std::variant` 방문이나 `constexpr` 테이블로 처리한다.

## 3.6 `mul_mat` 커널의 흐름 (CPU)

1. `src0`(가중치, 예: Q4_0)의 `vec_dot_type`(Q8_0)을 조회한다.
2. `src1`(활성값, F32)이 `vec_dot_type`이 아니면 **작업 버퍼에 양자화 사본**을 만든다. 이 단계도 스레드로 나눈다.
3. 배리어 후, `(src0 행, src1 열)` 타일을 청크로 나누고 각 스레드가 `current_chunk`를 가져가며 `vec_dot(n, &dst, src0_row, src1_col)`을 반복한다. 아키텍처별 `vec_dot_q4_0_q8_0`는 `src/ggml-cpu/arch/x86/quants.c` 등에 있다.
4. `GGML_CPU_REPACK`이 켜져 있고 가중치 버퍼가 "extra" 버퍼 타입이면, 3.5의 `ggml_cpu_extra_compute_forward`가 인터리브 레이아웃용 GEMM(`repack.cpp`)으로 가로챈다.
5. `GGML_LLAMAFILE` 또는 BLAS가 켜져 있으면 F32/F16 큰 행렬은 `llamafile/sgemm.cpp` 또는 BLAS로 우회한다.

이 흐름은 "양자화 = 저장 포맷"이 아니라 "양자화 = 커널 입력 포맷"임을 보여 준다. 새 프로젝트에서 Q4_0×Q8_0 정수 내적을 먼저 구현하면 llama.cpp 성능의 핵심을 재현한 것이다.

## 3.7 이 장의 체크포인트

- [ ] `ggml_mul_mat(A, B)`의 결과 `ne`를 `A.ne`, `B.ne`로 쓸 수 있다.
- [ ] `nodes[]`가 위상 정렬돼 있어서 실행기가 단순 루프로 충분한 이유를 설명할 수 있다.
- [ ] 노드 사이 배리어와 `mul_mat`의 동적 청크가 각각 어떤 문제를 푸는지 말할 수 있다.
- [ ] `vec_dot_type`이 무엇이고 왜 작업 버퍼가 필요한지 설명할 수 있다.
