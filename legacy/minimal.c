/*
 * Minimal end-to-end check for the virtual NPU matmul paths.
 *
 * One tiny matmul (4x8x4) sent to:
 *   - CPU reference
 *   - NPU CPU path
 *   - NPU Voodoo path
 *   - NPU Wildcat path
 *
 * Each NPU result compared against the CPU reference.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "virtual_npu.h"

static float *alloc_f(size_t n)
{
        return (float *)calloc(n, 1);
}

static void cpu_matmul(int M, int K, int N,
                     const float *A, const float *B, float *C)
{
        for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                        float acc = 0.0f;
                        for (int k = 0; k < K; k++) {
                                acc += A[m * K + k] * B[k * N + n];
                        }
                        C[m * N + n] = acc;
                }
        }
}

static int npu_matmul(virtual_npu_t *n,
                    int M, int K, int N,
                    const float *A_host, const float *B_host,
                    float *C_host,
                    size_t tile_m, size_t tile_k, size_t tile_n,
                    uint32_t op_kind, uint32_t op_code,
                    uint32_t cull_threshold)
{
        size_t a_bytes = (size_t)M * K * sizeof(float);
        size_t b_bytes = (size_t)K * N * sizeof(float);
        size_t c_bytes = (size_t)M * N * sizeof(float);

        size_t vram = virtual_npu_memory_size(n);
        if (a_bytes + b_bytes + c_bytes > vram)
                return -1;

        uint8_t *vram_mem = virtual_npu_memory(n);
        size_t a_addr = 0;
        size_t b_addr = a_bytes;
        size_t c_addr = a_bytes + b_bytes;
        size_t ctl_addr = a_bytes + b_bytes + c_bytes;

        memcpy(vram_mem + a_addr, A_host, a_bytes);
        memcpy(vram_mem + b_addr, B_host, b_bytes);
        memset(vram_mem + c_addr, 0, c_bytes);

        if (cull_threshold != 0 && ctl_addr + (size_t)M * N * 4 > virtual_npu_memory_size(n))
                return -1;

        if (cull_threshold != 0) {
                for (size_t i = 0; i < (size_t)M * N; i++)
                        ((float *)vram_mem)[ctl_addr / 4 + i] =
                                (float)((A_host[i % ((size_t)M * K)] > 0.02f &&
                                        B_host[i % ((size_t)K * N)] > 0.02f))
                                        ? 1.0f
                                        : 0.0f;
        }

        virtual_npu_compute(n,
                            a_addr,
                            b_addr,
                            c_addr,
                            ctl_addr,
                            (size_t)M,
                            (size_t)K,
                            (size_t)N,
                            tile_m,
                            tile_k,
                            tile_n,
                            op_kind,
                            op_code,
                            cull_threshold);
        virtual_npu_process(n);

        memcpy(C_host, vram_mem + c_addr, c_bytes);
        return 0;
}

static int compare_float(const float *a, const float *b, size_t n, float tol)
{
        for (size_t i = 0; i < n; i++) {
                float d = a[i] - b[i];
                if (d < 0)
                        d = -d;
                if (d > tol) {
                        printf("  first mismatch @%zu cpu=%g npu=%g\n", i, a[i], b[i]);
                        return 1;
                }
        }
        return 0;
}

int main(void)
{
        int M = 4;
        int K = 8;
        int N = 4;

        virtual_npu_t *n = virtual_npu_create(1 * 1024 * 1024);
        if (!n) {
                fprintf(stderr, "failed to create virtual npu\n");
                return 1;
        }

        float *A = alloc_f(M * K);
        float *B = alloc_f(K * N);
        float *C_cpu = alloc_f(M * N);
        float *C_cpu2 = alloc_f(M * N);
        float *C_voodoo = alloc_f(M * N);
        float *C_wildcat = alloc_f(M * N);
        if (!A || !B || !C_cpu || !C_cpu2 || !C_voodoo || !C_wildcat) {
                fprintf(stderr, "alloc failed\n");
                return 1;
        }

        for (size_t i = 0; i < (size_t)M * K; i++)
                A[i] = (float)((i & 0xff) + 1) * 0.001f;
        for (size_t i = 0; i < (size_t)K * N; i++)
                B[i] = (float)((i & 0xff) + 1) * 0.001f;

        cpu_matmul(M, K, N, A, B, C_cpu);

        fprintf(stderr, "[minimal] vram %zu bytes\n",
                virtual_npu_memory_size(n));

        size_t tile_m = 4;
        size_t tile_k = 8;
        size_t tile_n = 4;
        uint32_t cull = 0;

        int ok = 1;

        {
                virtual_npu_reset(n);
                if (npu_matmul(n, M, K, N, A, B, C_cpu2,
                                tile_m, tile_k, tile_n,
                                NPU_OP_CPU, 0x01, cull)) {
                        fprintf(stderr, "NPU CPU FAILED\n");
                        ok = 0;
                } else if (compare_float(C_cpu, C_cpu2, (size_t)M * N, 1e-2f)) {
                        fprintf(stderr, "NPU CPU MISMATCH\n");
                        ok = 0;
                } else {
                        printf("NPU CPU OK\n");
                }
        }

        {
                virtual_npu_reset(n);
                if (npu_matmul(n, M, K, N, A, B, C_voodoo,
                                tile_m, tile_k, tile_n,
                                NPU_OP_VOODOO, 0x01, cull)) {
                        fprintf(stderr, "NPU Voodoo FAILED\n");
                        ok = 0;
                } else if (compare_float(C_cpu, C_voodoo, (size_t)M * N, 1e-2f)) {
                        fprintf(stderr, "NPU Voodoo MISMATCH\n");
                        ok = 0;
                } else {
                        printf("NPU Voodoo OK\n");
                }
        }

        {
                virtual_npu_reset(n);
                if (npu_matmul(n, M, K, N, A, B, C_wildcat,
                                tile_m, tile_k, tile_n,
                                NPU_OP_WILDCAT, 0x01, cull)) {
                        fprintf(stderr, "NPU Wildcat FAILED\n");
                        ok = 0;
                } else if (compare_float(C_cpu, C_wildcat, (size_t)M * N, 1e-2f)) {
                        fprintf(stderr, "NPU Wildcat MISMATCH\n");
                        ok = 0;
                } else {
                        printf("NPU Wildcat OK\n");
                }
        }

        virtual_npu_destroy(n);
        free((void *)A);
        free((void *)B);
        free((void *)C_cpu);
        free((void *)C_cpu2);
        free((void *)C_voodoo);
        free((void *)C_wildcat);

        return ok ? 0 : 1;
}
