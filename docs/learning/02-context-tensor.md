# 2. 컨텍스트, 텐서, 메모리 레이아웃

## 2.1 `ggml_context`: 고정 크기 아레나

```c
// src/ggml.c:975
struct ggml_context {
    size_t mem_size;
    void * mem_buffer;
    bool   mem_buffer_owned;
    bool   no_alloc;
    int    n_objects;
    struct ggml_object * objects_begin;
    struct ggml_object * objects_end;
};
```

- `ggml_init(params)`는 `mem_size` 바이트짜리 버퍼 하나를 확보한다(`mem_buffer == NULL`이면 내부에서 `malloc`, 아니면 사용자 버퍼 사용). 이후 **이 컨텍스트에서 만들어지는 모든 것**(텐서 헤더, 텐서 데이터, 그래프)은 이 버퍼 안에 순차 배치된다.
- 각 객체 앞에는 `ggml_object` 헤더(`src/ggml.c:958`)가 붙어 연결 리스트를 이룬다. 타입은 `TENSOR`, `GRAPH`, `WORK_BUFFER`.
- 버퍼가 차면 `ggml_new_object`가 실패(assert)한다. 그래서 예제들은 크기를 미리 계산한다:

```cpp
// examples/simple/simple-ctx.cpp:24-31
ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F32); // tensor a 데이터
ctx_size += 2 * ggml_tensor_overhead();                       // 텐서 헤더 2개
ctx_size += ggml_graph_overhead();                            // 그래프
ctx_size += 1024;                                             // 여유
```

- `no_alloc = true`면 텐서 **헤더만** 컨텍스트에 만들고 `data`는 `NULL`로 둔다. 데이터는 나중에 백엔드 버퍼에 할당한다. 백엔드 API를 쓸 때의 표준 패턴이다(`simple-backend.cpp:69-73`: `ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead()`).

C++23 관점: 이것은 `std::pmr::monotonic_buffer_resource` 위에 놓인 타입별 리스트다. 다만 ggml은 **텐서 헤더와 데이터를 같은 버퍼에 두는 것**과 **`ggml_reset()`으로 통째로 되감는 것**에 의존하므로, 단순 교체보다 의미를 먼저 이해해야 한다.

## 2.2 `ggml_tensor`: 메타데이터 + 그래프 노드

```c
// include/ggml.h:685
struct ggml_tensor {
    enum ggml_type type;
    struct ggml_backend_buffer * buffer;   // 데이터가 놓인 백엔드 버퍼 (없으면 NULL)
    int64_t ne[GGML_MAX_DIMS];             // 원소 수 (GGML_MAX_DIMS = 4)
    size_t  nb[GGML_MAX_DIMS];             // 바이트 스트라이드
    enum ggml_op op;                       // 이 텐서를 만든 연산 (리프면 GGML_OP_NONE)
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)]; // 64바이트 연산 인자
    int32_t flags;                         // INPUT/OUTPUT/PARAM/LOSS/COMPUTE
    struct ggml_tensor * src[GGML_MAX_SRC]; // 입력 (최대 10개)
    struct ggml_tensor * view_src;         // 뷰라면 원본
    size_t               view_offs;        // 원본으로부터의 바이트 오프셋
    void * data;
    char name[GGML_MAX_NAME];              // 64바이트
    void * extra;                          // 백엔드 전용 (예: CUDA split 정보)
    char padding[8];
};
```

한 구조체가 **세 역할**을 동시에 한다.
1. 배열 뷰: `type`, `ne`, `nb`, `data`
2. 그래프 노드: `op`, `op_params`, `src[]`, `flags`
3. 할당 단위: `buffer`, `view_src`, `view_offs`

C++23으로 옮길 때 가장 먼저 부딪히는 설계 질문이 "이 셋을 분리할 것인가"다. 분리하면 타입 안전성이 오르지만, ggml의 할당기·스케줄러는 `view_src`와 `src[]`를 같은 자료구조에서 순회하는 데 의존한다(4장 참조). 7장에서 다시 논의한다.

## 2.3 `ne`와 `nb`: 차원 순서를 반드시 외울 것

**`ne[0]`가 가장 안쪽(연속) 차원이다.** NumPy/PyTorch의 shape를 뒤집은 순서다.

```
PyTorch  x.shape == (batch, heads, seq, dim)
ggml     ne == { dim, seq, heads, batch }
```

스트라이드는 생성 시 자동 계산된다:

```c
// src/ggml.c:1833
result->nb[0] = ggml_type_size(type);
result->nb[1] = result->nb[0]*(result->ne[0]/ggml_blck_size(type));
for (int i = 2; i < GGML_MAX_DIMS; i++) result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
```

`nb[1]`에 `ggml_blck_size(type)`로 나누는 부분이 양자화의 핵심이다. Q4_0은 32개 원소가 18바이트 블록 하나이므로 `ne[0]`는 32의 배수여야 하고, 행 하나의 바이트 수는 `ne[0]/32 * 18`이다. `ggml_row_size(type, ne0)`가 이 계산을 캡슐화한다.

원소 `(i0, i1, i2, i3)`의 주소는 항상
```
(char*)data + i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3]
```
이다. 커널은 이 공식을 매크로 `GGML_TENSOR_UNARY_OP_LOCALS` 등으로 펼쳐 쓴다(`src/ggml-cpu/ops.cpp:56`).

C++23 대응: `std::mdspan<T, std::dextents<int64_t, 4>, std::layout_stride>`가 `ne`/`nb`를 정확히 표현한다. 단 `nb`가 **바이트** 단위이고 양자화 타입은 원소 단위 접근이 불가능하므로, `mdspan`은 F32/F16 같은 비양자화 타입에만 그대로 쓰고 양자화 타입은 "블록의 mdspan"으로 모델링해야 한다.

## 2.4 뷰: 복사 없는 재해석

`ggml_view_2d`, `ggml_permute`, `ggml_transpose`, `ggml_reshape_*`는 모두 새 텐서 헤더를 만들되 `data`를 원본 안으로 가리키게 한다.

```c
// src/ggml.c:1778 — 뷰의 뷰는 원본까지 따라 올라가 절대 오프셋으로 정규화
if (view_src != NULL && view_src->view_src != NULL) {
    view_offs += view_src->view_offs;
    view_src   = view_src->view_src;
}
```

- `ggml_permute(ctx, a, 1, 0, 2, 3)`는 `ne`/`nb`를 자리바꿈만 한다. 결과는 **비연속(non-contiguous)** 텐서다.
- 대부분의 커널은 `src`가 비연속이어도 `nb`로 접근하므로 동작하지만, 일부(`mul_mat`의 `src0` 등)는 연속을 요구한다. 그때 `ggml_cont()`를 끼워 넣으면 실제 복사 연산 노드(`GGML_OP_CONT`)가 생긴다.
- `ggml_is_contiguous(t)`가 이를 판별한다.

할당기 관점에서 뷰는 메모리를 차지하지 않는다(`ggml-alloc.c:627`에서 `ggml_impl_is_view(node)`면 건너뜀). 대신 원본의 생명주기를 연장시킨다(`n_views` 카운트).

## 2.5 타입 시스템

```c
// include/ggml.h:389
enum ggml_type { GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1, GGML_TYPE_Q4_0 = 2, ... GGML_TYPE_NVFP4 = 40, GGML_TYPE_Q1_0 = 41, GGML_TYPE_Q2_0 = 42, GGML_TYPE_COUNT = 43 };
```

- enum 값은 **GGUF 파일에 그대로 기록**되므로 재사용되지 않는다. 삭제된 타입(4, 5, 31-33, 36-38)은 주석으로 남아 있다. 새 프로젝트도 이 번호를 그대로 써야 GGUF 호환이 된다.
- 각 타입의 속성은 `ggml_get_type_traits(type)`(`include/ggml.h:2976`)이 돌려주는 테이블에 있다:

```c
struct ggml_type_traits {
    const char             * type_name;
    int64_t                  blck_size;             // 블록당 원소 수 (F32=1, Q4_0=32, Q4_K=256)
    int64_t                  blck_size_interleave;
    size_t                   type_size;             // 블록 바이트 수
    bool                     is_quantized;
    ggml_to_float_t          to_float;              // 역양자화
    ggml_from_float_t        from_float_ref;        // 양자화 참조 구현
};
```

CPU 백엔드는 여기에 `vec_dot`, `vec_dot_type`, `nrows` 등을 더한 자체 테이블(`src/ggml-cpu/ggml-cpu.c:219-269`)을 가진다. 예: Q4_0 가중치의 `vec_dot_type`은 `Q8_0`이다. 즉 F32 활성값을 Q8_0으로 양자화한 뒤 Q4_0×Q8_0 정수 내적을 한다(3장).

C++23 대응: `consteval`로 채우는 `constexpr std::array<type_traits, 43>` + `enum class ggml_type : int32_t`. `std::to_underlying`으로 파일 포맷 값과 매핑한다.

## 2.6 `flags`

```c
// include/ggml.h:663
GGML_TENSOR_FLAG_INPUT   = 1,  // 그래프 입력: 할당기가 그래프 시작 부분에 겹치지 않게 배치
GGML_TENSOR_FLAG_OUTPUT  = 2,  // 그래프 출력: 절대 해제/덮어쓰기하지 않음
GGML_TENSOR_FLAG_PARAM   = 4,  // 학습 파라미터
GGML_TENSOR_FLAG_LOSS    = 8,  // 손실
GGML_TENSOR_FLAG_COMPUTE = 16, // 계산 필요 (역전파 그래프에서 사용)
```

`ggml_set_input()`/`ggml_set_output()`을 빼먹으면 할당기가 중간 텐서 메모리를 재사용해 결과가 덮어써진다. 초보자가 가장 자주 겪는 버그다.

## 2.7 실습 포인터

- `examples/simple/simple-ctx.cpp`를 읽고 `ggml_used_mem(ctx)`를 출력해 예약한 크기와 비교하라.
- `ggml_permute` 후 `ne`/`nb`를 출력해 `nb[0] != ggml_type_size(type)`이 되는 것을 확인하라 (8장 실습 2).

## 이 장의 체크포인트

- [ ] `ne = {2, 4}`인 F32 텐서의 `nb`를 손으로 계산할 수 있다 (정답: `{4, 8, 32, 32}`).
- [ ] `ne0 = 64`인 Q4_0 텐서 한 행이 몇 바이트인지 계산할 수 있다 (정답: `64/32 * 18 = 36`).
- [ ] `no_alloc = true`일 때 `data`가 언제 채워지는지 말할 수 있다.
