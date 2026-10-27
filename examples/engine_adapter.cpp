/*
 * engine_adapter.cpp - framework-neutral GGML/llama.cpp-style backend sketch.
 *
 * This file is documentation-quality code: it deliberately avoids including
 * ggml.h or any model runtime headers.  Copy the mapping into a real backend
 * plugin and replace EngineTensor with your engine's tensor type.
 */
#include "vnpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace example {

struct EngineTensor {
    const char *name;
    float *data;
    int64_t ne[4];
    size_t nb[4];
};

struct Session {
    vnpu_device_t *device = nullptr;
    vnpu_stream_t *stream = nullptr;
    vnpu_event_t *event = nullptr;
    vnpu_packed_weights_t *wqkv = nullptr;
    vnpu_packed_weights_t *wo = nullptr;
    vnpu_packed_weights_t *wgateup = nullptr;
    vnpu_packed_weights_t *wdown = nullptr;
};

static vnpu_status_t init(Session *s)
{
    vnpu_config_t cfg;
    vnpu_config_init(&cfg);
    cfg.flags = VNPU_CONFIG_HIGH_THROUGHPUT;
    vnpu_status_t rc = vnpu_device_create(&cfg, &s->device);
    if (rc != VNPU_OK) return rc;
    rc = vnpu_stream_create(s->device, 0, &s->stream);
    if (rc != VNPU_OK) return rc;
    return vnpu_event_create(&s->event);
}

static void destroy(Session *s)
{
    vnpu_packed_weights_destroy(s->wqkv);
    vnpu_packed_weights_destroy(s->wo);
    vnpu_packed_weights_destroy(s->wgateup);
    vnpu_packed_weights_destroy(s->wdown);
    vnpu_event_destroy(s->event);
    vnpu_stream_destroy(s->stream);
    vnpu_device_destroy(s->device);
}

static vnpu_status_t pack_weight(Session *s, const EngineTensor *w,
                                  vnpu_packed_weights_t **out)
{
    /* In a real adapter, quantize once at model load and keep the int8 buffer.
     * Here we assume w->data is already int8 reinterpreted as float storage. */
    const int8_t *bytes = reinterpret_cast<const int8_t *>(w->data);
    uint32_t K = static_cast<uint32_t>(w->ne[0]);
    uint32_t N = static_cast<uint32_t>(w->ne[1]);
    return vnpu_pack_weights_i8(s->device, bytes, nullptr, K, N, 0, 0, out);
}

static vnpu_status_t build_layer_commands(
    Session *s,
    std::vector<vnpu_command_t> &cmds,
    const EngineTensor *x,
    const EngineTensor *norm_w,
    const EngineTensor *qkv_out,
    const EngineTensor *attn_out,
    const EngineTensor *ffn_out,
    uint32_t tokens,
    uint32_t hidden,
    uint32_t intermediate,
    uint32_t heads,
    uint32_t seq_len)
{
    cmds.clear();
    cmds.reserve(12);
    const uint32_t head_dim = hidden / heads;
    const uint32_t qkv_dim = 3 * hidden;

    /* 1. RMSNorm. */
    {
        vnpu_command_t c{};
        c.opcode = VNPU_OP_RMSNORM_F32;
        c.args.rmsnorm.X = x->data;
        c.args.rmsnorm.W = norm_w->data;
        c.args.rmsnorm.Y = x->data;  /* in-place if engine allows it */
        c.args.rmsnorm.rows = tokens;
        c.args.rmsnorm.cols = hidden;
        c.args.rmsnorm.eps = 1e-5f;
        cmds.push_back(c);
    }
    /* 2. QKV projection (INT8 packed weights). */
    {
        vnpu_command_t c{};
        c.opcode = VNPU_OP_MATMUL_I8;
        c.args.matmul_i8.A = reinterpret_cast<const int8_t *>(x->data);
        c.args.matmul_i8.B = s->wqkv;
        c.args.matmul_i8.C = qkv_out->data;
        c.args.matmul_i8.M = tokens;
        c.args.matmul_i8.K = hidden;
        c.args.matmul_i8.N = qkv_dim;
        c.args.matmul_i8.unit = tokens > 1 ? VNPU_UNIT_VOODOO :
                                              VNPU_UNIT_CELL;
        cmds.push_back(c);
    }
    /* 3. RoPE on Q and K slices. */
    {
        vnpu_command_t c{};
        c.opcode = VNPU_OP_ROPE_F32;
        c.args.rope.Q = qkv_out->data;
        c.args.rope.K = qkv_out->data + hidden;
        c.args.rope.batch = 1;
        c.args.rope.heads = heads;
        c.args.rope.q_seq = seq_len;
        c.args.rope.k_seq = seq_len;
        c.args.rope.head_dim = head_dim;
        c.args.rope.theta_base = 10000.0f;
        c.args.rope.scaling = 1.0f;
        cmds.push_back(c);
    }
    /* 4. Fused attention. */
    {
        vnpu_command_t c{};
        c.opcode = VNPU_OP_ATTENTION_F32;
        c.args.attention.Q = qkv_out->data;
        c.args.attention.K = qkv_out->data + hidden;
        c.args.attention.V = qkv_out->data + 2 * hidden;
        c.args.attention.O = attn_out->data;
        c.args.attention.batch = 1;
        c.args.attention.heads = heads;
        c.args.attention.seq_q = seq_len;
        c.args.attention.seq_k = seq_len;
        c.args.attention.head_dim = head_dim;
        c.args.attention.causal = 1;
        c.args.attention.scale = 1.0f /
            static_cast<float>(std::sqrt((double)head_dim));
        cmds.push_back(c);
    }
    /* 5. Output projection. */
    {
        vnpu_command_t c{};
        c.opcode = VNPU_OP_MATMUL_I8;
        c.args.matmul_i8.A =
            reinterpret_cast<const int8_t *>(attn_out->data);
        c.args.matmul_i8.B = s->wo;
        c.args.matmul_i8.C = ffn_out->data;
        c.args.matmul_i8.M = tokens;
        c.args.matmul_i8.K = hidden;
        c.args.matmul_i8.N = hidden;
        cmds.push_back(c);
    }
    /* 6. Residual + second RMSNorm + gate/up + SwiGLU + down would follow the
     * same pattern.  The point is to submit all of them in ONE batch. */
    (void)intermediate;
    return VNPU_OK;
}

static vnpu_status_t run_layer(Session *s,
                               std::vector<vnpu_command_t> &cmds,
                               uint32_t timeout_ms = 30000)
{
    vnpu_status_t rc = vnpu_stream_submit(s->stream, cmds.data(),
                                           cmds.size(), s->event);
    if (rc != VNPU_OK) return rc;
    return vnpu_event_wait(s->event, timeout_ms);
}

} /* namespace example */

#if defined(VNPU_ENGINE_ADAPTER_DEMO)
int main()
{
    using namespace example;
    Session s;
    vnpu_status_t rc = init(&s);
    if (rc != VNPU_OK) {
        std::fprintf(stderr, "init failed: %s\n", vnpu_status_string(rc));
        return 1;
    }
    std::vector<vnpu_command_t> cmds;
    /* Fill EngineTensor descriptors from a real model here. */
    destroy(&s);
    return 0;
}
#endif
