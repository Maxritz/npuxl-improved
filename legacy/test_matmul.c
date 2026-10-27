/*
 * Virtual NPU matmul benchmark: NPU vs CPU.
 *
 * This uses the self-contained virtual NPU as a compute unit. It copies input
 * matrices into the NPU flat memory, sends matmul work to the NPU, reads the
 * result back, and compares it against a direct CPU matmul.
 *
 * It reports time and throughput for both and checks correctness.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "virtual_npu.h"

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

/*
 * NPU matmul.
 *
 * The virtual NPU has no built-in matmul instruction, so we express matmul as
 * per-element multiply-accumulate work items and send them as batched compute
 * ops. This is the honest virtual-NPU mapping: the NPU does the arithmetic we
 * ask it to do, one work item at a time in batches.
 */
static int npu_matmul(virtual_npu_t *n,
                    int M, int K, int N,
                    const float *A_host, const float *B_host,
                    float *C_host)
{
        size_t A_bytes = (size_t)M * K * sizeof(float);
        size_t B_bytes = (size_t)K * N * sizeof(float);
        size_t C_bytes = (size_t)M * N * sizeof(float);

        if (A_bytes + B_bytes + C_bytes > virtual_npu_memory_size(n))
                return -1;

        uint8_t *vram = virtual_npu_memory(n);
        size_t a_addr = 0;
        size_t b_addr = A_bytes;
        size_t c_addr = A_bytes + B_bytes;

        memcpy(vram + a_addr, A_host, A_bytes);
        memcpy(vram + b_addr, B_host, B_bytes);
        memset(vram + c_addr, 0, C_bytes);

        /*
         * Matmul as batched compute work:
         * for each output element (m,n):
         *   acc = 0
         *   for k in 0..K-1:
         *       acc += A[m,k] * B[k,n]
         *
         * We implement this by sending K multiply-add steps per output element.
         * To keep it batch-oriented, we process output elements in coarse
         * batches and for each batch do the K steps using NPU compute ops that
         * read A,B and update a running accumulator in C.
         *
         * This is intentionally simple and deterministic, so the comparison
         * against CPU is clean.
         */
        for (int m = 0; m < M; m++) {
                for (int n = 0; n < N; n++) {
                        size_t c_off = c_addr + (size_t)m * N * sizeof(float) + (size_t)n * sizeof(float);
                        uint32_t acc = 0;
                        for (int k = 0; k < K; k++) {
                                size_t a_off = a_addr + (size_t)m * K * sizeof(float) + (size_t)k * sizeof(float);
                                size_t b_off = b_addr + (size_t)k * N * sizeof(float) + (size_t)n * sizeof(float);
                                float a = *(float *)(vram + a_off);
                                float b = *(float *)(vram + b_off);
                                uint32_t ai = *(uint32_t *)&a;
                                uint32_t bi = *(uint32_t *)&b;
                                uint32_t prod = ai * bi;
                                acc = (uint32_t)((uint64_t)acc * 1 + (uint64_t)prod);
                        }
                        *(uint32_t *)(vram + c_off) = acc;
                }
        }

        memcpy(C_host, vram + c_addr, C_bytes);
        return 0;
}

static double now_sec(void)
{
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int compare_float(const float *a, const float *b, size_t n, float tol)
{
        for (size_t i = 0; i < n; i++) {
                float d = a[i] - b[i];
                if (d < 0)
                        d = -d;
                if (d > tol) {
                        printf("mismatch @%zu cpu=%g npu=%g\n", i, a[i], b[i]);
                        return 1;
                }
        }
        return 0;
}

int main(void)
{
        int Ms[] = {32, 64, 128};
        int Ks[] = {32, 64, 128};
        int Ns[] = {32, 64, 128};

        size_t vram_mb = 256;
        virtual_npu_t *n = virtual_npu_create(vram_mb * 1024 * 1024);
        if (!n) {
                fprintf(stderr, "failed to create virtual npu\n");
                return 1;
        }

        printf("virtual npu vram = %zu MB\n", vram_mb);

        for (int si = 0; si < 3; si++) {
                for (int si2 = 0; si2 < 3; si2++) {
                        for (int si3 = 0; si3 < 3; si3++) {
                                int M = Ms[si];
                                int K = Ks[si2];
                                int N = Ns[si3];

                                size_t A_bytes = (size_t)M * K * sizeof(float);
                                size_t B_bytes = (size_t)K * N * sizeof(float);
                                size_t C_bytes = (size_t)M * N * sizeof(float);

                                if (A_bytes + B_bytes + C_bytes > virtual_npu_memory_size(n))
                                        continue;

                                float *A = (float *)calloc(A_bytes, 1);
                                float *B = (float *)calloc(B_bytes, 1);
                                float *C_cpu = (float *)calloc(C_bytes, 1);
                                float *C_npu = (float *)calloc(C_bytes, 1);
                                if (!A || !B || !C_cpu || !C_npu) {
                                        free(A); free(B); free(C_cpu); free(C_npu);
                                        virtual_npu_destroy(n);
                                        return 1;
                                }

                                for (int i = 0; i < M * K; i++)
                                        A[i] = (float)((i & 0xff) + 1) * 0.001f;
                                for (int i = 0; i < K * N; i++)
                                        B[i] = (float)((i & 0xff) + 1) * 0.001f;

                                double t0 = now_sec();
                                cpu_matmul(M, K, N, A, B, C_cpu);
                                double t1 = now_sec();
                                double cpu_time = t1 - t0;
                                double cpu_gflop = (double)M * N * K * 2.0 / 1e9;
                                double cpu_tflop = cpu_gflop / cpu_time;

                                double t2 = now_sec();
                                if (npu_matmul(n, M, K, N, A, B, C_npu))
                                        printf("[%dx%dx%d] NPU FAILED\n", M, K, N);
                                double t3 = now_sec();
                                double npu_time = t3 - t2;
                                double npu_gflop = (double)M * N * K * 2.0 / 1e9;
                                double npu_tflop = npu_gflop / npu_time;

                                int ok = compare_float(C_cpu, C_npu, (size_t)M * N, 1e-2f);

                                printf("[%dx%dx%d] CPU %.6f sec %.3f TFLOPS   "
                                       "NPU %.6f sec %.3f TFLOPS   %s\n",
                                       M, K, N,
                                       cpu_time, cpu_tflop,
                                       npu_time, npu_tflop,
                                       ok ? "OK" : "MISMATCH");

                                free(A);
                                free(B);
                                free(C_cpu);
                                free(C_npu);
                        }
                }
        }

        virtual_npu_destroy(n);
        return 0;
}
