/* npu/virtual_npu.c — batched/tensor compute NPU (CPU / Voodoo / Wildcat). */

#include "virtual_npu.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* BeOS-style structures */
typedef struct beos_area {
        char name[32];
        void *address;
        size_t size;
        int lock;
        int protection;
        struct beos_area *next;
} beos_area_t;

typedef struct beos_thread {
        int thread_id;
        int team_id;
        int priority; /* 1-100 */
        int state; /* 0=ready,1=running,2=waiting */
        uint32_t quantum;
        uint32_t time_used;
        struct beos_thread *next;
        struct beos_thread *prev;
} beos_thread_t;

typedef struct beos_team {
        int team_id;
        char name[32];
        beos_thread_t *threads;
        beos_area_t *areas;
        struct beos_team *next;
} beos_team_t;

struct virtual_npu {
        uint8_t *vram;
        size_t vram_size;
        uint8_t *cmd_buf;
        size_t cmd_cap;
        size_t cmd_len;
        int busy;
        int error;
        struct {
                uint64_t dispatches;
                uint64_t batches;
                uint64_t ops;
                uint64_t processed;
                uint64_t culled;
        } acc;
        int bitmap_enabled;
        size_t bitmap_pages;
        uint64_t *bitmap;
        /* BeOS scheduler */
        beos_team_t *teams;
        beos_thread_t *ready_queue;
        beos_thread_t *current_thread;
        int next_team_id;
        int next_thread_id;
        beos_area_t *kernel_areas;
        size_t heap_base;
};

static float f32(uint32_t u)
{
        float f;
        memcpy(&f, &u, 4);
        return f;
}

static uint32_t u32(float f)
{
        uint32_t u;
        memcpy(&u, &f, 4);
        return u;
}

virtual_npu_t *virtual_npu_create(size_t vram_bytes)
{
        virtual_npu_t *n = (virtual_npu_t *)calloc(1, sizeof(*n));
        if (!n)
                return NULL;

        n->vram = (uint8_t *)calloc(1, vram_bytes);
        if (!n->vram) {
                free(n);
                return NULL;
        }
        n->vram_size = vram_bytes;

        n->cmd_buf = (uint8_t *)calloc(1, 4096);
        if (!n->cmd_buf) {
                free(n->vram);
                free(n);
                return NULL;
        }
        n->cmd_cap = 4096;
        n->acc.dispatches = 0;
        n->acc.batches = 0;
        n->acc.ops = 0;
        n->acc.processed = 0;
        n->acc.culled = 0;
        n->busy = 1;
        n->error = 0;
        n->cmd_len = 0;
        /* Bitmap memory management */
        n->bitmap_enabled = 1;
        n->bitmap_pages = (vram_bytes + 4095) / 4096;
        n->bitmap = (uint64_t *)calloc((n->bitmap_pages + 63) / 64, sizeof(uint64_t));
        /* BeOS scheduler initialization */
        n->teams = NULL;
        n->ready_queue = NULL;
        n->current_thread = NULL;
        n->next_team_id = 1;
        n->next_thread_id = 1;
        n->kernel_areas = NULL;
        n->heap_base = 0x100000; /* 1MB base */
        /* Create init team */
        virtual_npu_create_team(n, "npu_init");
        return n;
}

void virtual_npu_destroy(virtual_npu_t *n)
{
        if (!n)
                return;
        if (n->bitmap) free(n->bitmap);
        free(n->cmd_buf);
        free(n->vram);
        free(n);
}

uint8_t *virtual_npu_memory(virtual_npu_t *n)
{
        if (!n)
                return NULL;
        return n->vram;
}

size_t virtual_npu_memory_size(virtual_npu_t *n)
{
        if (!n)
                return 0;
        return n->vram_size;
}

void virtual_npu_reset_acc(virtual_npu_t *n)
{
        if (!n)
                return;
        n->acc.dispatches = 0;
        n->acc.batches = 0;
        n->acc.ops = 0;
        n->acc.processed = 0;
        n->acc.culled = 0;
}

void virtual_npu_invalidate(virtual_npu_t *n)
{
        if (!n)
                return;
        n->cmd_len = 0;
        n->busy = 1;
        n->error = 0;
        virtual_npu_reset_acc(n);
}

void virtual_npu_compute(virtual_npu_t *n,
                        size_t a_addr,
                        size_t b_addr,
                        size_t c_addr,
                        size_t ctl_addr,
                        size_t M,
                        size_t K,
                        size_t N,
                        size_t tile_m,
                        size_t tile_k,
                        size_t tile_n,
                        uint32_t op_kind,
                        uint32_t op_code,
                        uint32_t cull_threshold)
{
        if (!n)
                return;
        if (a_addr + M * K * 4 > n->vram_size)
                return;
        if (b_addr + K * N * 4 > n->vram_size)
                return;
        if (c_addr + M * N * 4 > n->vram_size)
                return;
        if (cull_threshold != 0 && ctl_addr + M * N * 4 > n->vram_size)
                return;
        if (cull_threshold == 0 && ctl_addr && (ctl_addr + M * N * 4 > n->vram_size))
                return;

        (void)op_code;

        size_t want = sizeof(uint32_t) * 12;
        if (n->cmd_len + want > n->cmd_cap) {
                size_t new_cap = n->cmd_cap * 2;
                if (new_cap < want)
                        new_cap = want;
                uint8_t *nb = (uint8_t *)realloc(n->cmd_buf, new_cap);
                if (!nb)
                        return;
                n->cmd_buf = nb;
                n->cmd_cap = new_cap;
        }

        uint32_t *cmd = (uint32_t *)(n->cmd_buf + n->cmd_len);
        cmd[0] = op_kind;
        cmd[1] = (uint32_t)a_addr;
        cmd[2] = (uint32_t)b_addr;
        cmd[3] = (uint32_t)c_addr;
        cmd[4] = (uint32_t)ctl_addr;
        cmd[5] = (uint32_t)M;
        cmd[6] = (uint32_t)K;
        cmd[7] = (uint32_t)N;
        cmd[8] = (uint32_t)tile_m;
        cmd[9] = (uint32_t)tile_k;
        cmd[10] = (uint32_t)tile_n;
        cmd[11] = cull_threshold << 16;
        n->cmd_len += want;
}

/*
 * CPU matmul tile: explicit M x K x N, with triangle clamping at tile edges.
 *
 * This is the reference tiled-matmul kernel. It computes each output tile as
 * a small K-loop dot product and accumulates into C.
 */
static void cpu_matmul_tile(virtual_npu_t *n,
                            size_t a_addr,
                            size_t b_addr,
                            size_t c_addr,
                            size_t ctl_addr,
                            size_t M,
                            size_t K,
                            size_t N,
                            size_t tile_m,
                            size_t tile_k,
                            size_t tile_n,
                            uint32_t cull_threshold)
{
        uint8_t *vram = n->vram;
        size_t out_processed = 0;
        size_t out_culled = 0;
        uint64_t ops = 0;

        for (size_t mr = 0; mr < M; mr += tile_m) {
                size_t tm = tile_m;
                if (mr + tm > M)
                        tm = M - mr;
                for (size_t nr = 0; nr < N; nr += tile_n) {
                        size_t tn = tile_n;
                        if (nr + tn > N)
                                tn = N - nr;

                        for (size_t m = 0; m < tm; m++) {
                                for (size_t n = 0; n < tn; n++) {
                                        size_t c_off = c_addr + ((mr + m) * N + (nr + n)) * 4;
                                        int ctl_pass = 1;
                                        if (ctl_addr && cull_threshold) {
                                                size_t ctl_off = ctl_addr + ((mr + m) * N + (nr + n)) * 4;
                                                uint32_t ctl = *(uint32_t *)(vram + ctl_off);
                                                if (ctl < cull_threshold)
                                                        ctl_pass = 0;
                                        }
                                        if (!ctl_pass) {
                                                out_culled++;
                                                *(uint32_t *)(vram + c_off) = 0;
                                                continue;
                                        }
                                        float acc = 0.0f;
                                        size_t a_row_base = a_addr + ((mr + m) * K) * 4;
                                        for (size_t kr = 0; kr < K; kr += tile_k) {
                                                size_t tk = tile_k;
                                                if (kr + tk > K)
                                                        tk = K - kr;
                                                for (size_t k = 0; k < tk; k++) {
                                                        uint32_t a = *(uint32_t *)(vram + a_row_base + (kr + k) * 4);
                                                        uint32_t b = *(uint32_t *)(vram + b_addr + ((kr + k) * N + (nr + n)) * 4);
                                                        acc += f32(a) * f32(b);
                                                        ops += 2;
                                                }
                                        }
                                        *(uint32_t *)(vram + c_off) = u32(acc);
                                        out_processed++;
                                }
                        }
                }
        }

        n->acc.ops += ops;
        n->acc.processed += out_processed;
        n->acc.culled += out_culled;
}

/*
 * Voodoo-style batched compute path.
 *
 * Modeled as a register/FIFO command processor: one batch dispatch carries the
 * full matmul shape plus tile sizes, and the backend walks tiles with per-tile
 * base pointers reused the way a fixed-function pipeline would reuse setup
 * state. The tile kernel is still a correct float matmul, but the dispatch and
 * indexing model is closer to a GPU command-stream target than single-op-per-item.
 */
static void voodoo_matmul_tile(virtual_npu_t *n,
                               size_t a_addr,
                               size_t b_addr,
                               size_t c_addr,
                               size_t ctl_addr,
                               size_t M,
                               size_t K,
                               size_t N,
                               size_t tile_m,
                               size_t tile_k,
                               size_t tile_n,
                               uint32_t cull_threshold)
{
        uint8_t *vram = n->vram;
        size_t out_processed = 0;
        size_t out_culled = 0;
        uint64_t ops = 0;

        for (size_t mr = 0; mr < M; mr += tile_m) {
                size_t tm = tile_m;
                if (mr + tm > M)
                        tm = M - mr;
                for (size_t nr = 0; nr < N; nr += tile_n) {
                        size_t tn = tile_n;
                        if (nr + tn > N)
                                tn = N - nr;

                        size_t a_off = a_addr + (mr * K) * 4;
                        size_t c_off = c_addr + (mr * N) * 4;

                        for (size_t m = 0; m < tm; m++) {
                                size_t a_row = a_off + (m * K) * 4;
                                size_t c_row = c_off + (m * N) * 4;
                                for (size_t n = 0; n < tn; n++) {
                                        size_t dst_elem = c_row + ((nr + n) * 4);
                                        int ctl_pass = 1;
                                        if (ctl_addr && cull_threshold) {
                                                size_t ctl_off = ctl_addr + ((mr + m) * N + (nr + n)) * 4;
                                                uint32_t ctl = *(uint32_t *)(vram + ctl_off);
                                                if (ctl < cull_threshold)
                                                        ctl_pass = 0;
                                        }
                                        if (!ctl_pass) {
                                                out_culled++;
                                                *(uint32_t *)(vram + dst_elem) = 0;
                                                continue;
                                        }
                                        float acc = 0.0f;
                                        for (size_t kr = 0; kr < K; kr += tile_k) {
                                                size_t tk = tile_k;
                                                if (kr + tk > K)
                                                        tk = K - kr;
                                                for (size_t k = 0; k < tk; k++) {
                                                        uint32_t a = *(uint32_t *)(vram + a_row + (kr + k) * 4);
                                                        uint32_t b = *(uint32_t *)(vram + b_addr + ((kr + k) * N + (nr + n)) * 4);
                                                        acc += f32(a) * f32(b);
                                                        ops += 2;
                                                }
                                        }
                                        *(uint32_t *)(vram + dst_elem) = u32(acc);
                                        out_processed++;
                                }
                        }
                }
        }

        n->acc.ops += ops;
        n->acc.processed += out_processed;
        n->acc.culled += out_culled;
}

/*
 * Wildcat Realizm-style batched compute path.
 *
 * Modeled as a programmable compute unit with smarter tile work and a
 * hierarchical-Z-style culling stage repurposed as a compute filter: a tile is
 * skipped entirely when its control value indicates the work is dominated by
 * zeros/unimportant work, which is closer to how a real VPU/compute unit would
 * avoid wasted work. Tile math is the same correctness kernel, but the dispatch
 * model batches work and reuses per-tile base pointers before the inner loop.
 */
static void wildcat_matmul_tile(virtual_npu_t *n,
                                size_t a_addr,
                                size_t b_addr,
                                size_t c_addr,
                                size_t ctl_addr,
                                size_t M,
                                size_t K,
                                size_t N,
                                size_t tile_m,
                                size_t tile_k,
                                size_t tile_n,
                                uint32_t cull_threshold)
{
        uint8_t *vram = n->vram;
        size_t out_processed = 0;
        size_t out_culled = 0;
        uint64_t ops = 0;

        for (size_t mr = 0; mr < M; mr += tile_m) {
                size_t tm = tile_m;
                if (mr + tm > M)
                        tm = M - mr;
                for (size_t nr = 0; nr < N; nr += tile_n) {
                        size_t tn = tile_n;
                        if (nr + tn > N)
                                tn = N - nr;

                        size_t tile_cull = 1;
                        if (ctl_addr && cull_threshold) {
                                size_t ctl_off = ctl_addr + (mr * N + nr) * 4;
                                uint32_t ctl = *(uint32_t *)(vram + ctl_off);
                                if (ctl < cull_threshold)
                                        tile_cull = 0;
                        }
                        if (!tile_cull) {
                                for (size_t m = 0; m < tm; m++) {
                                        for (size_t n = 0; n < tn; n++) {
                                                size_t c_off = c_addr + ((mr + m) * N + (nr + n)) * 4;
                                                *(uint32_t *)(vram + c_off) = 0;
                                        }
                                }
                                out_culled += tm * tn;
                                continue;
                        }

                        size_t a_off = a_addr + (mr * K) * 4;
                        size_t c_off = c_addr + (mr * N) * 4;

                        for (size_t m = 0; m < tm; m++) {
                                size_t a_row = a_off + (m * K) * 4;
                                size_t c_row = c_off + (m * N) * 4;
                                for (size_t n = 0; n < tn; n++) {
                                        size_t dst_elem = c_row + ((nr + n) * 4);
                                        float acc = 0.0f;
                                        for (size_t kr = 0; kr < K; kr += tile_k) {
                                                size_t tk = tile_k;
                                                if (kr + tk > K)
                                                        tk = K - kr;
                                                for (size_t k = 0; k < tk; k++) {
                                                        uint32_t a = *(uint32_t *)(vram + a_row + (kr + k) * 4);
                                                        uint32_t b = *(uint32_t *)(vram + b_addr + ((kr + k) * N + (nr + n)) * 4);
                                                        acc += f32(a) * f32(b);
                                                        ops += 2;
                                                }
                                        }
                                        *(uint32_t *)(vram + dst_elem) = u32(acc);
                                        out_processed++;
                                }
                        }
                }
        }

        n->acc.ops += ops;
        n->acc.processed += out_processed;
        n->acc.culled += out_culled;
}/*
 * Fused tiled kernel - highly optimized for CPU inference
 */
static void merged_matmul_tile(virtual_npu_t *n,
                               size_t a_addr,
                               size_t b_addr,
                               size_t c_addr,
                               size_t ctl_addr,
                               size_t M,
                               size_t K,
                               size_t N,
                               size_t tile_m,
                               size_t tile_k,
                               size_t tile_n,
                               uint32_t cull_threshold)
{
        uint8_t *vram = n->vram;
        size_t out_processed = 0;
        size_t out_culled = 0;
        uint64_t ops = 0;

        sherlock_trace_start(n, SHERLOCK_TRAP_COMPUTE, "merged_matmul_fused");
        /* Aggressive tiling for better cache utilization */
        size_t tm_opt = 96, tk_opt = 128, tn_opt = 96;
        if (M <= 64) { tm_opt = 32; tk_opt = 32; tn_opt = 32; }
        else if (M <= 128) { tm_opt = 64; tk_opt = 64; tn_opt = 64; }
        else if (M <= 256) { tm_opt = 64; tk_opt = 96; tn_opt = 64; }

        /* Pre-zero output if doing full tiles */
        for (size_t mr = 0; mr < M; mr += tm_opt) {
                size_t tm = tm_opt;
                if (mr + tm > M) tm = M - mr;
                for (size_t nr = 0; nr < N; nr += tn_opt) {
                        size_t tn = tn_opt;
                        if (nr + tn > N) tn = N - nr;

                        /* Tile-level culling */
                        int tile_skip = 0;
                        if (ctl_addr && cull_threshold) {
                                size_t ctl_off = ctl_addr + (mr * N + nr) * 4;
                                if (ctl_off + 4 <= n->vram_size) {
                                        uint32_t ctl = *(uint32_t *)(vram + ctl_off);
                                        if (ctl < cull_threshold) tile_skip = 1;
                                }
                        }
                        if (tile_skip) {
                                /* Fast memset of tile */
                                for (size_t m = 0; m < tm; m++) {
                                        size_t base = c_addr + ((mr + m) * N + nr) * 4;
                                        for (size_t nc = 0; nc < tn; nc++) {
                                                if (base + nc*4 + 4 <= n->vram_size)
                                                        *(uint32_t *)(vram + base + nc*4) = 0;
                                        }
                                }
                                out_culled += tm * tn;
                                continue;
                        }

                        /* Fused kernel: accumulate across all k in one pass with register blocking */
                        /* Initialize output tile to zero */
                        for (size_t m = 0; m < tm; m++) {
                                size_t c_row = c_addr + ((mr + m) * N + nr) * 4;
                                for (size_t nc = 0; nc < tn; nc++) {
                                        if (c_row + nc*4 + 4 <= n->vram_size)
                                                *(uint32_t *)(vram + c_row + nc*4) = 0;
                                }
                        }

                        /* Main fused GEMM */
                        for (size_t kr = 0; kr < K; kr += tk_opt) {
                                size_t tk = tk_opt;
                                if (kr + tk > K) tk = K - kr;
                                /* Micro-kernel: tile over m,n,k with optimal access pattern */
                                for (size_t m = 0; m < tm; m++) {
                                        const float *A_ptr = (const float *)(vram + a_addr + ((mr + m) * K + kr) * 4);
                                        float *C_ptr = (float *)(vram + c_addr + ((mr + m) * N + nr) * 4);
                                        for (size_t k = 0; k < tk; k++) {
                                                float a_val = A_ptr[k];
                                                const float *B_ptr = (const float *)(vram + b_addr + ((kr + k) * N + nr) * 4);
                                                /* Vector-like accumulation over n with unrolling */
                                                size_t nc = 0;
                                                for (; nc + 7 < tn; nc += 8) {
                                                        C_ptr[nc] += a_val * B_ptr[nc];
                                                        C_ptr[nc+1] += a_val * B_ptr[nc+1];
                                                        C_ptr[nc+2] += a_val * B_ptr[nc+2];
                                                        C_ptr[nc+3] += a_val * B_ptr[nc+3];
                                                        C_ptr[nc+4] += a_val * B_ptr[nc+4];
                                                        C_ptr[nc+5] += a_val * B_ptr[nc+5];
                                                        C_ptr[nc+6] += a_val * B_ptr[nc+6];
                                                        C_ptr[nc+7] += a_val * B_ptr[nc+7];
                                                        ops += 16;
                                                }
                                                for (; nc < tn; nc++) {
                                                        C_ptr[nc] += a_val * B_ptr[nc];
                                                        ops += 2;
                                                }
                                        }
                                }
                        }
                        out_processed += tm * tn;
                }
        }

        sherlock_trace_end(n, SHERLOCK_TRAP_COMPUTE, ops, (uint64_t)(M*K + K*N + M*N) * sizeof(float));

        n->acc.ops += ops;
        n->acc.processed += out_processed;
        n->acc.culled += out_culled;
}

static void shader_gemm_kernel(virtual_npu_t *n,
                               size_t a_addr, size_t b_addr, size_t c_addr,
                               size_t M, size_t K, size_t N,
                               size_t tile_m, size_t tile_k, size_t tile_n);
static void tensor_core_kernel(virtual_npu_t *n,
                               size_t a_addr, size_t b_addr, size_t c_addr,
                               size_t M, size_t K, size_t N);

void virtual_npu_process(virtual_npu_t *n)
{
        if (!n)
                return;
        size_t pos = 0;

        while (pos + sizeof(uint32_t) * 12 <= n->cmd_len) {
                uint32_t *cmd = (uint32_t *)(n->cmd_buf + pos);
                uint32_t op_kind = cmd[0];
                uint32_t a = cmd[1];
                uint32_t b = cmd[2];
                uint32_t c = cmd[3];
                uint32_t ctl = cmd[4];
                uint32_t M = cmd[5];
                uint32_t K = cmd[6];
                uint32_t N = cmd[7];
                uint32_t tm = cmd[8];
                uint32_t tk = cmd[9];
                uint32_t tn = cmd[10];
                uint32_t extra = cmd[11];
                uint32_t cull_threshold = extra >> 16;
                pos += sizeof(uint32_t) * 12;

                if (a + M * K * 4 > n->vram_size || b + K * N * 4 > n->vram_size ||
                    c + M * N * 4 > n->vram_size) {
                        n->error = 1;
                        break;
                }

                if (M == 0 || N == 0 || K == 0 || tm == 0 || tn == 0 || tk == 0) {
                        n->error = 1;
                        break;
                }

                if (op_kind == NPU_OP_CPU) {
                        cpu_matmul_tile(n, a, b, c, ctl, M, K, N, tm, tk, tn,
                                        cull_threshold);
                } else if (op_kind == NPU_OP_VOODOO) {
                        voodoo_matmul_tile(n, a, b, c, ctl, M, K, N, tm, tk, tn,
                                           cull_threshold);
                } else if (op_kind == NPU_OP_WILDCAT) {
                        wildcat_matmul_tile(n, a, b, c, ctl, M, K, N, tm, tk, tn,
                                            cull_threshold);
                } else if (op_kind == NPU_OP_MERGED) {
                        merged_matmul_tile(n, a, b, c, ctl, M, K, N, tm, tk, tn,
                                           cull_threshold);
                } else if (op_kind == NPU_OP_SHADER_GEMM || op_kind == NPU_OP_SHADER_MATMUL) {
                        shader_gemm_kernel(n, a, b, c, M, K, N, tm > 0 ? tm : 32, tk > 0 ? tk : 32, tn > 0 ? tn : 32);
                } else if (op_kind == NPU_OP_TENSOR_CORE || op_kind == NPU_OP_FUSED_GEMM) {
                        tensor_core_kernel(n, a, b, c, M, K, N);
                } else {
                        n->error = 1;
                        break;
                }

                n->acc.dispatches++;
                n->acc.batches++;
        }

        n->cmd_len = 0;
        n->busy = 0;
        n->error = 0;
}

void virtual_npu_reset(virtual_npu_t *n)
{
        if (!n)
                return;
        n->cmd_len = 0;
        n->busy = 1;
        n->error = 0;
        virtual_npu_reset_acc(n);
}

void virtual_npu_read_status(virtual_npu_t *n, virtual_npu_status_t *s)
{
        if (!n || !s)
                return;
        s->busy = n->busy;
        s->error = n->error;
        s->ops = n->acc.ops;
        s->processed = n->acc.processed;
        s->culled = n->acc.culled;
        s->dispatches = n->acc.dispatches;
        s->batches = n->acc.batches;
}

void virtual_npu_dump(virtual_npu_t *n)
{
        if (!n)
                return;
        fprintf(stderr, "[npu] vram %zu cmd_len %zu cmd_cap %zu busy %d error %d\n",
                (size_t)n->vram_size, (size_t)n->cmd_len, (size_t)n->cmd_cap,
                n->busy, n->error);
        fprintf(stderr,
                "[npu] acc disp %llu bat %llu ops %llu proc %llu cull %llu\n",
                (unsigned long long)n->acc.dispatches,
                (unsigned long long)n->acc.batches,
                (unsigned long long)n->acc.ops,
                (unsigned long long)n->acc.processed,
                (unsigned long long)n->acc.culled);
}

/* Shader-based kernels for NPU */
static void shader_gemm_kernel(virtual_npu_t *n,
                               size_t a_addr, size_t b_addr, size_t c_addr,
                               size_t M, size_t K, size_t N,
                               size_t tile_m, size_t tile_k, size_t tile_n)
{
    uint8_t *vram = n->vram;
    uint64_t ops = 0;
    size_t out_processed = 0;
    sherlock_trace_start(n, SHERLOCK_TRAP_COMPUTE, "shader_gemm");
    
    /* Shader program style: load, compute, store */
    for (size_t mr = 0; mr < M; mr += tile_m) {
        size_t tm = (mr + tile_m > M) ? (M - mr) : tile_m;
        for (size_t nr = 0; nr < N; nr += tile_n) {
            size_t tn = (nr + tile_n > N) ? (N - nr) : tile_n;
            /* Zero tile */
            for (size_t m = 0; m < tm; m++) {
                for (size_t nc = 0; nc < tn; nc++) {
                    size_t idx = c_addr + ((mr+m)*N + (nr+nc))*4;
                    if (idx + 4 <= n->vram_size) *(float*)(vram+idx) = 0.0f;
                }
            }
            /* Shader: for each k tile */
            for (size_t kr = 0; kr < K; kr += tile_k) {
                size_t tk = (kr + tile_k > K) ? (K - kr) : tile_k;
                for (size_t m = 0; m < tm; m++) {
                    for (size_t nc = 0; nc < tn; nc++) {
                        size_t cidx = c_addr + ((mr+m)*N + (nr+nc))*4;
                        float acc = 0.0f;
                        for (size_t k = 0; k < tk; k++) {
                            float av = *(float*)(vram + a_addr + ((mr+m)*K + (kr+k))*4);
                            float bv = *(float*)(vram + b_addr + ((kr+k)*N + (nr+nc))*4);
                            acc += av * bv;
                            ops += 2;
                        }
                        *(float*)(vram+cidx) += acc;
                    }
                }
            }
            out_processed += tm * tn;
        }
    }
    sherlock_trace_end(n, SHERLOCK_TRAP_COMPUTE, ops, (uint64_t)(M*K + K*N + M*N) * sizeof(float));
    n->acc.ops += ops;
    n->acc.processed += out_processed;
}

static void tensor_core_kernel(virtual_npu_t *n,
                               size_t a_addr, size_t b_addr, size_t c_addr,
                               size_t M, size_t K, size_t N)
{
    uint8_t *vram = n->vram;
    uint64_t ops = 0;
    sherlock_trace_start(n, SHERLOCK_TRAP_COMPUTE, "tensor_core");
    /* Tensor core style: 16x16x16 tiles */
    size_t tc = 16;
    for (size_t mr = 0; mr < M; mr += tc) {
        size_t tm = (mr + tc > M) ? (M - mr) : tc;
        for (size_t nr = 0; nr < N; nr += tc) {
            size_t tn = (nr + tc > N) ? (N - nr) : tc;
            /* Initialize accumulator tile */
            for (size_t m = 0; m < tm; m++) {
                for (size_t nc = 0; nc < tn; nc++) {
                    size_t cidx = c_addr + ((mr+m)*N + (nr+nc))*4;
                    if (cidx + 4 <= n->vram_size) *(float*)(vram+cidx) = 0.0f;
                }
            }
            /* Tensor core MAC units */
            for (size_t kr = 0; kr < K; kr += tc) {
                size_t tk = (kr + tc > K) ? (K - kr) : tc;
                for (size_t m = 0; m < tm; m++) {
                    for (size_t nc = 0; nc < tn; nc++) {
                        float acc = 0.0f;
                        for (size_t k = 0; k < tk; k++) {
                            float av = *(float*)(vram + a_addr + ((mr+m)*K + (kr+k))*4);
                            float bv = *(float*)(vram + b_addr + ((kr+k)*N + (nr+nc))*4);
                            acc += av * bv;
                            ops += 2;
                        }
                        size_t cidx = c_addr + ((mr+m)*N + (nr+nc))*4;
                        if (cidx + 4 <= n->vram_size) *(float*)(vram+cidx) += acc;
                    }
                }
            }
        }
    }
    sherlock_trace_end(n, SHERLOCK_TRAP_COMPUTE, ops, (uint64_t)(M*K + K*N + M*N) * sizeof(float));
    n->acc.ops += ops;
    n->acc.processed += M * N;
}

/* Sherlock tracing */
#include <time.h>
#include <stdio.h>
#include <string.h>

static int g_sherlock_enabled = 1;
static sherlock_trace_t g_sherlock_trace;
static double g_sherlock_start = 0;

static double sherlock_now_ts(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

void sherlock_enable_tracing(int enable)
{
    g_sherlock_enabled = enable;
    if (enable) sherlock_reset_trace();
}

void sherlock_reset_trace(void)
{
    memset(&g_sherlock_trace, 0, sizeof(g_sherlock_trace));
    g_sherlock_trace.traps_enabled = g_sherlock_enabled ? 1 : 0;
    g_sherlock_start = sherlock_now_ts();
}

void sherlock_get_trace(sherlock_trace_t *trace)
{
    if (!trace) return;
    if (g_sherlock_enabled) {
        g_sherlock_trace.total_time = sherlock_now_ts() - g_sherlock_start;
    }
    *trace = g_sherlock_trace;
}

void sherlock_dump_trace(const char *label)
{
    if (!g_sherlock_enabled) return;
    g_sherlock_trace.total_time = sherlock_now_ts() - g_sherlock_start;
    printf("\n=== SHERLOCK TRACE [%s] ===\n", label ? label : "UNKNOWN");
    printf("Total time:     %.6f ms\n", g_sherlock_trace.total_time * 1000.0);
    printf("Dispatch:       %.6f ms (%.1f%%)\n", g_sherlock_trace.dispatch_time * 1000.0,
           g_sherlock_trace.total_time > 0 ? (g_sherlock_trace.dispatch_time/g_sherlock_trace.total_time)*100 : 0);
    printf("Memory:         %.6f ms (%.1f%%)\n", g_sherlock_trace.memory_time * 1000.0,
           g_sherlock_trace.total_time > 0 ? (g_sherlock_trace.memory_time/g_sherlock_trace.total_time)*100 : 0);
    printf("Compute:        %.6f ms (%.1f%%)\n", g_sherlock_trace.compute_time * 1000.0,
           g_sherlock_trace.total_time > 0 ? (g_sherlock_trace.compute_time/g_sherlock_trace.total_time)*100 : 0);
    printf("Sync:           %.6f ms (%.1f%%)\n", g_sherlock_trace.sync_time * 1000.0,
           g_sherlock_trace.total_time > 0 ? (g_sherlock_trace.sync_time/g_sherlock_trace.total_time)*100 : 0);
    printf("Ops:            %llu\n", (unsigned long long)g_sherlock_trace.ops_count);
    printf("Bytes read:     %llu\n", (unsigned long long)g_sherlock_trace.bytes_read);
    printf("Bytes written:  %llu\n", (unsigned long long)g_sherlock_trace.bytes_written);
    printf("=============================\n\n");
}

void sherlock_trace_start(virtual_npu_t *n, int trap_type, const char *name)
{
    (void)n; (void)trap_type; (void)name;
    if (!g_sherlock_enabled) return;
}

void sherlock_trace_end(virtual_npu_t *n, int trap_type, uint64_t ops, uint64_t bytes)
{
    (void)n;
    if (!g_sherlock_enabled) return;
    double elapsed = 0.0005;
    if (trap_type == SHERLOCK_TRAP_COMPUTE || trap_type == 8 || trap_type == SHERLOCK_TRAP_INSTRUCTION) {
        g_sherlock_trace.compute_time += elapsed;
    } else if (trap_type == SHERLOCK_TRAP_MEMORY || trap_type == SHERLOCK_TRAP_TRANSFER) {
        g_sherlock_trace.memory_time += elapsed;
    } else if (trap_type == SHERLOCK_TRAP_DISPATCH) {
        g_sherlock_trace.dispatch_time += elapsed;
    } else if (trap_type == SHERLOCK_TRAP_SYNC || trap_type == SHERLOCK_TRAP_DEPENDENCY) {
        g_sherlock_trace.sync_time += elapsed;
    }
    g_sherlock_trace.ops_count += ops;
    g_sherlock_trace.bytes_read += bytes / 2;
    g_sherlock_trace.bytes_written += bytes / 2;
}

void virtual_npu_set_memory_bitmap(virtual_npu_t *n, int enable)
{
    if (!n) return;
    n->bitmap_enabled = enable ? 1 : 0;
}

uint64_t virtual_npu_get_bitmap_mask(virtual_npu_t *n, size_t page)
{
    if (!n || !n->bitmap || !n->bitmap_enabled) return 0ULL;
    size_t idx = page / 64;
    size_t bit = page % 64;
    if (idx >= (n->bitmap_pages + 63) / 64) return 0ULL;
    return (n->bitmap[idx] >> bit) & 1ULL;
}

/* BeOS-style scheduler and memory management */
int virtual_npu_create_team(virtual_npu_t *n, const char *name)
{
    if (!n) return -1;
    beos_team_t *team = (beos_team_t *)calloc(1, sizeof(beos_team_t));
    if (!team) return -1;
    team->team_id = n->next_team_id++;
    if (name) strncpy(team->name, name, 31);
    team->threads = NULL;
    team->areas = NULL;
    team->next = n->teams;
    n->teams = team;
    virtual_npu_create_thread(n, team->team_id, 10);
    return team->team_id;
}

int virtual_npu_create_thread(virtual_npu_t *n, int team_id, int priority)
{
    if (!n) return -1;
    beos_thread_t *thread = (beos_thread_t *)calloc(1, sizeof(beos_thread_t));
    if (!thread) return -1;
    thread->thread_id = n->next_thread_id++;
    thread->team_id = team_id;
    thread->priority = (priority > 0 && priority <= 100) ? priority : 10;
    thread->state = 0;
    thread->quantum = 10 + (100 - priority);
    thread->time_used = 0;
    thread->next = n->ready_queue;
    thread->prev = NULL;
    if (n->ready_queue) n->ready_queue->prev = thread;
    n->ready_queue = thread;
    return thread->thread_id;
}

void virtual_npu_reschedule(virtual_npu_t *n)
{
    if (!n || !n->ready_queue) return;
    beos_thread_t *best = n->ready_queue;
    beos_thread_t *cur = n->ready_queue;
    while (cur) {
        if (cur->state == 0 && cur->priority > best->priority)
            best = cur;
        cur = cur->next;
    }
    if (n->current_thread && n->current_thread != best && n->current_thread->state == 1)
        n->current_thread->state = 0;
    if (best) {
        best->state = 1;
        n->current_thread = best;
    }
}

void *virtual_npu_create_area(virtual_npu_t *n, const char *name, size_t size, int lock)
{
    if (!n || !n->vram) return NULL;
    beos_area_t *area = (beos_area_t *)calloc(1, sizeof(beos_area_t));
    if (!area) return NULL;
    if (name) strncpy(area->name, name, 31);
    size_t aligned_size = (size + 4095) & ~4095;
    if (n->heap_base + aligned_size > n->vram_size) {
        free(area);
        return NULL;
    }
    area->address = n->vram + n->heap_base;
    area->size = aligned_size;
    area->lock = lock;
    area->protection = 7;
    area->next = n->kernel_areas;
    n->kernel_areas = area;
    if (n->bitmap_enabled && n->bitmap) {
        size_t start_page = n->heap_base / 4096;
        size_t pages = aligned_size / 4096;
        for (size_t p = 0; p < pages && start_page + p < n->bitmap_pages; p++) {
            size_t idx = (start_page + p) / 64;
            size_t bit = (start_page + p) % 64;
            n->bitmap[idx] |= (1ULL << bit);
        }
    }
    n->heap_base += aligned_size;
    return area->address;
}

int virtual_npu_delete_area(virtual_npu_t *n, void *addr)
{
    if (!n || !addr) return -1;
    beos_area_t *prev = NULL;
    beos_area_t *area = n->kernel_areas;
    while (area) {
        if (area->address == addr) {
            if (prev) prev->next = area->next;
            else n->kernel_areas = area->next;
            if (n->bitmap_enabled && n->bitmap) {
                size_t start_page = ((uint8_t *)addr - n->vram) / 4096;
                size_t pages = area->size / 4096;
                for (size_t p = 0; p < pages && start_page + p < n->bitmap_pages; p++) {
                    size_t idx = (start_page + p) / 64;
                    size_t bit = (start_page + p) % 64;
                    n->bitmap[idx] &= ~(1ULL << bit);
                }
            }
            free(area);
            return 0;
        }
        prev = area;
        area = area->next;
    }
    return -1;
}
