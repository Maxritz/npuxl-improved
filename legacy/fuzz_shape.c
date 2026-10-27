/*
 * Small correctness fuzz for the virtual NPU matmul paths.
 *
 * For many random M x K x N shapes within the NPU vram budget, run:
 *   - CPU matmul reference
 *   - NPU CPU path
 *   - NPU Voodoo path
 *   - NPU Wildcat path
 * and check each against the CPU reference.
 *
 * If anything mismatches or the NPU reports an error, dump state and abort.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "virtual_npu.h"

static float *alloc_f(size_t n)
{
        return (float *)calloc(n, 1);
}

static float randf(void)
{
        return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
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
        srand((unsigned)time(NULL));

        size_t vram_mb = 256;
        virtual_npu_t *n = virtual_npu_create(vram_mb * 1024 * 1024);
        if (!n) {
                fprintf(stderr, "failed to create virtual npu\n");
                return 1;
        }

        fprintf(stderr, "[fuzz] vram %zu MB, starting fuzz with up to 2000 shapes\n",
                vram_mb);

        for (int iter = 0; iter < 2000; iter++) {
                int M = 2 + (rand() % 120);
                int K = 2 + (rand() % 120);
                int N = 2 + (rand() % 120);
                int cull = (rand() & 1);

                size_t a_bytes = (size_t)M * K * sizeof(float);
                size_t b_bytes = (size_t)K * N * sizeof(float);
                size_t c_bytes = (size_t)M * N * sizeof(float);
                if (a_bytes + b_bytes + c_bytes > virtual_npu_memory_size(n))
                        continue;

                float *A = alloc_f(M * K);
                float *B = alloc_f(K * N);
                float *C_cpu = alloc_f(M * N);
                float *C_cpu2 = alloc_f(M * N);
                float *C_voodoo = alloc_f(M * N);
                float *C_wildcat = alloc_f(M * N);
                if (!A || !B || !C_cpu || !C_cpu2 || !C_voodoo || !C_wildcat) {
                        free(A); free(B); free(C_cpu); free(C_cpu2);
                        free(C_voodoo); free(C_wildcat);
                        virtual_npu_destroy(n);
                        return 1;
                }

                for (size_t i = 0; i < (size_t)M * K; i++)
                        A[i] = randf();
                for (size_t i = 0; i < (size_t)K * N; i++)
                        B[i] = randf();

                cpu_matmul(M, K, N, A, B, C_cpu);

                int rc = 0;

                {
                        virtual_npu_invalidate(n);
                        virtual_npu_reset_acc(n);
                        if (npu_matmul(n, M, K, N, A, B, C_cpu2,
                                        4, 8, 4,
                                        NPU_OP_CPU, 0x01, cull)) {
                                rc = -1;
                                fprintf(stderr, "NPU CPU FAILED\n");
                        } else if (compare_float(C_cpu, C_cpu2, (size_t)M * N, 1e-2f)) {
                                rc = -2;
                        }
                }

                if (!rc) {
                        virtual_npu_invalidate(n);
                        virtual_npu_reset_acc(n);
                        if (npu_matmul(n, M, K, N, A, B, C_voodoo,
                                        4, 8, 4,
                                        NPU_OP_VOODOO, 0x01, cull)) {
                                rc = -3;
                                fprintf(stderr, "NPU Voodoo FAILED\n");
                        } else if (compare_float(C_cpu, C_voodoo, (size_t)M * N, 1e-2f)) {
                                rc = -4;
                        }
                }

                if (!rc) {
                        virtual_npu_invalidate(n);
                        virtual_npu_reset_acc(n);
                        if (npu_matmul(n, M, K, N, A, B, C_wildcat,
                                        4, 8, 4,
                                        NPU_OP_WILDCAT, 0x01, cull)) {
                                rc = -5;
                                fprintf(stderr, "NPU Wildcat FAILED\n");
                        } else if (compare_float(C_cpu, C_wildcat, (size_t)M * N, 1e-2f)) {
                                rc = -6;
                        }
                }

                if (rc) {
                        fprintf(stderr, "FAIL iter=%d M=%d K=%d N=%d cull=%d rc=%d\n",
                                iter, M, K, N, cull, rc);
                        virtual_npu_dump(n);
                        fprintf(stderr, "  A[%d,%d] B[%d,%d] C[%d,%d] cull=%d\n",
                                M, K, K, N, M, N, cull);
                        free(A); free(B); free(C_cpu); free(C_cpu2);
                        free(C_voodoo); free(C_wildcat);
                        virtual_npu_destroy(n);
                        return 1;
                }

                free(A); free(B); free(C_cpu); free(C_cpu2);
                free(C_voodoo); free(C_wildcat);
        }

        virtual_npu_destroy(n);
        printf("OK: %d random matmul shapes passed\n", 2000);
        return 0;
}
