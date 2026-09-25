# 5. 양자화 포맷과 GGUF

## 5.1 블록 양자화의 기본형: Q4_0

```c
// src/ggml-common.h:194-199
#define QK4_0 32
typedef struct {
    ggml_half d;           // delta (scale), F16
    uint8_t qs[QK4_0 / 2]; // 4비트 값 32개 → 16바이트
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, "wrong q4_0 block size/padding");
```

- 32개 F32 원소 → 18바이트. 원소당 4.5비트.
- 역양자화: `x[i] = d * (q[i] - 8)` (부호 없는 4비트에서 8을 뺀 대칭 양자화).
- `qs`의 니블 배치는 `qs[j]`의 하위 4비트가 원소 `j`, 상위 4비트가 원소 `j+16`이다. SIMD에서 두 절반을 한 번에 풀기 위한 배치이며, **다른 구현과 호환하려면 이 순서를 지켜야 한다**.

타입 계열:
| 계열 | 블록 크기 | 특징 |
|---|---|---|
| Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 | 32 | 단순 블록. `_1`은 min 오프셋 추가 |
| Q2_K ... Q8_K | 256 | "K-quant". 슈퍼블록 안에 서브블록별 6비트 스케일. llama.cpp 기본 |
| IQ1_S ... IQ4_XS | 32/256 | 코드북(격자) 기반. 일부는 importance matrix 필수(`ggml_quantize_requires_imatrix`) |
| TQ1_0, TQ2_0 | 256 | 삼진(ternary) |
| MXFP4, NVFP4 | 32/16 | 마이크로스케일링 FP4. `NVFP4`는 E4M3 스케일 |
| Q1_0, Q2_0 | - | 최근 추가 |

`ggml-common.h`의 `static_assert`는 **구조체 패딩이 없음을 보장**한다. 파일에 그대로 `memcpy`되는 구조체이므로 C++23에서도 `static_assert(sizeof(...) == ...)`와 `std::is_trivially_copyable_v`를 유지해야 한다.

## 5.2 양자화/역양자화 API

```c
// include/ggml.h:2946
size_t ggml_quantize_chunk(enum ggml_type type, const float * src, void * dst, int64_t start, int64_t nrows, int64_t n_per_row, const float * imatrix);
```

- 행 단위로 동작한다. `n_per_row`가 블록 크기의 배수여야 한다.
- 내부는 `ggml_get_type_traits(type)->from_float_ref` 또는 `quantize_<type>` (imatrix 지원 버전).
- 역양자화는 `traits->to_float(src, dst, n)`.
- `tests/test-quantize-fns.cpp`가 각 타입에 대해 (1) 양자화→역양자화 RMSE, (2) `vec_dot` 정확도를 참조 구현과 비교한다. 새 프로젝트의 **첫 번째 호환성 테스트**로 쓰기 좋다. `tests/test-quantize-perf.cpp`는 처리량을 잰다.

## 5.3 CPU 커널의 양자화 활용: `vec_dot_type`

3장에서 봤듯 CPU `mul_mat`은 활성값을 `vec_dot_type`으로 즉석 양자화한다. `src/ggml-cpu/ggml-cpu.c:219-269`의 테이블에서
- F32 → F32, F16 → F16
- Q4_0, Q5_0, Q8_0, Q2_K ... → **Q8_0** 또는 **Q8_1**(오프셋 있는 계열), K-quant → **Q8_K**

이유: 4비트×8비트 정수 내적은 `vpdpbusd`(AVX-VNNI), `sdot`(ARM dotprod) 같은 명령으로 매우 빠르다. F32×Q4_0을 직접 하면 매번 역양자화가 필요해 느리다.

새 프로젝트에서 성능 목표를 세울 때 "Q4_0 × Q8_0 vec_dot"의 아키텍처별 구현(`src/ggml-cpu/arch/x86/quants.c`, `arch/arm/quants.c`)이 기준선이다.

## 5.4 GGUF 파일 포맷

`include/gguf.h:1-30`의 주석이 스펙 요약이고 `docs/gguf.md`가 전체 스펙이다.

```
magic "GGUF" (4B) | version u32 (=3) | n_tensors i64 | n_kv i64
KV × n_kv:  key(string) | type(gguf_type i32) | value  (ARRAY면 elem type + count + elems)
TensorInfo × n_tensors: name(string) | n_dims u32 | ne[i] i64... | type(ggml_type i32) | offset u64
padding to general.alignment (기본 32)
tensor data blob (각 텐서는 offset 위치, alignment 정렬)
```

- 문자열은 `u64 길이 + 바이트`(널 종료 없음). 모든 정수는 리틀 엔디언.
- `general.alignment` KV가 있으면 그 값으로 정렬. 이 덕분에 **데이터 블롭을 통째로 `mmap`** 하고 각 텐서의 `data`를 오프셋 포인터로 잡을 수 있다(`buffer_from_host_ptr`).
- 메타데이터는 자유 키-값이다. `general.architecture`, `llama.context_length`, `tokenizer.ggml.tokens` 같은 키는 llama.cpp의 관례이며 ggml은 해석하지 않는다.
- API: `gguf_init_from_file(path, {no_alloc, &ctx})`. `ctx`를 넘기면 텐서 헤더가 담긴 `ggml_context`를 만들어 준다(`no_alloc=true`면 데이터 없이). 그 뒤 `gguf_get_tensor_offset`, `gguf_get_data_offset`으로 파일 안 위치를 얻는다.
- `gguf_reader_callback_t`(`gguf.h:80`)로 `FILE*` 대신 임의 소스(네트워크, 압축)에서 읽을 수 있다.

C++23 대응: 파서는 `std::span<const std::byte>` + `std::bit_cast` + `std::endian` 검사로 매우 깔끔해진다. `std::expected<gguf_file, parse_error>`로 오류를 돌려 준다. 쓰기는 `std::format`이 필요 없는 순수 바이너리이므로 `std::ofstream` 또는 `std::span`으로 충분하다.

## 5.5 반정밀도 타입

- `ggml_half`는 `uint16_t`이고 `ggml_fp16_to_fp32`/`ggml_fp32_to_fp16`가 변환한다. F16C/NEON이 있으면 하드웨어 변환, 없으면 비트 연산.
- `ggml_bf16_t`도 `uint16_t`. 상위 16비트 절단(반올림 포함).
- C++23은 `<stdfloat>`의 `std::float16_t`, `std::bfloat16_t`를 정의하지만 **선택 사항**이다. GCC 13+/Clang 15+는 x86/ARM에서 제공하지만 MSVC는 제공하지 않는다. vcpkg 삼중항(triplet)에 MSVC가 포함된다면 `uint16_t` 저장 + 명시적 변환 함수를 유지하는 편이 안전하다.

## 이 장의 체크포인트

- [ ] `block_q4_0` 18바이트의 구성을 그릴 수 있고, 니블 순서를 설명할 수 있다.
- [ ] Q4_0 가중치에 대해 `vec_dot_type`이 Q8_0인 이유를 SIMD 명령 관점에서 말할 수 있다.
- [ ] GGUF 헤더를 손으로 파싱해 첫 텐서의 데이터 오프셋을 구할 수 있다 (8장 실습 6).
