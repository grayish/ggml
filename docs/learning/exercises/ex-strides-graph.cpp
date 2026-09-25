#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <vector>

static void show(const char * tag, const ggml_tensor * t) {
    printf("%-8s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] contiguous=%d op=%s\n", tag,
        ggml_type_name(t->type), (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
        t->nb[0], t->nb[1], t->nb[2], t->nb[3], ggml_is_contiguous(t), ggml_op_name(t->op));
}

int main() {
    ggml_init_params p = { 16*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(p);

    // 실습 2: 스트라이드와 뷰
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 4);
    ggml_tensor * q = ggml_new_tensor_1d(ctx, GGML_TYPE_Q4_0, 64);
    ggml_tensor * at = ggml_transpose(ctx, a);
    ggml_tensor * ac = ggml_cont(ctx, at);
    show("a", a); show("q4_0", q); show("a^T", at); show("cont", ac);
    printf("row_size(Q4_0, 64) = %zu, used_mem = %zu\n", ggml_row_size(GGML_TYPE_Q4_0, 64), ggml_used_mem(ctx));

    // 실습 3: softmax(x W^T + b) 그래프
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2); // [in=3, batch=2]
    ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4); // [in=3, out=4]
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    for (int i = 0; i < 6;  ++i) ((float*)x->data)[i] = 0.1f * (i + 1);
    for (int i = 0; i < 12; ++i) ((float*)W->data)[i] = (i % 3 == 0) ? 1.0f : -0.5f;
    for (int i = 0; i < 4;  ++i) ((float*)b->data)[i] = 0.01f * i;
    ggml_tensor * y = ggml_soft_max(ctx, ggml_add(ctx, ggml_mul_mat(ctx, W, x), b));
    ggml_set_name(y, "y");
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_graph_print(gf);
    ggml_graph_compute_with_ctx(ctx, gf, 2);
    show("y", y);
    for (int r = 0; r < y->ne[1]; ++r) {
        printf("y[%d] =", r);
        for (int c = 0; c < y->ne[0]; ++c) printf(" %.4f", *(float*)((char*)y->data + r*y->nb[1] + c*y->nb[0]));
        printf("\n");
    }
    ggml_graph_dump_dot(gf, nullptr, "graph.dot");
    ggml_free(ctx);
}
