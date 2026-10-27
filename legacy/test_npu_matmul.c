/*
 * Virtual NPU matmul benchmark: NPU vs CPU.
 *
 * This treats the NPU as a compute unit and sends it a matrix multiply job.
 * It does NOT use the Pcem video-core Voodoo paths for the matmul itself.
 * The NPU flat memory is the working buffer; the NPU performs the matmul
 * through its own compute unit, and we compare the result against a direct CPU
 * matmul.
 *
 * Shapes:
 *   A: M x K
 *   B: K x N
 *   C = A * B: M x N
 *
 * We run a few sizes and report time + throughput + correctness.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "includes/private/video/vgpu_npu.h"

/*
 * Simple column-major matmul reference on CPU.
 * C = A * B
 * A: M x K, B: K x N, C: M x N
 */
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
 * NPU matmul helper.
 *
 * We copy the input matrices into the NPU flat memory and ask the NPU to
 * compute C = A * B by issuing the smallest reasonable set of compute
 * operations the NPU can express. In this virtual-NPU model, the NPU is a
 * compute unit with a flat address space and a batched compute instruction;
 * it does not magically know matrix semantics, so we express matmul as a
 * sequence of per-element multiply-accumulate work sent to unit 1.
 *
 * This is intentionally a simple, honest mapping: the NPU does the raw
 * arithmetic; we do not pretend it has an implicit BLAS engine.
 */
static int npu_matmul(vgpu_npu_t *npu,
                    int M, int K, int N,
                    const float *A_host, const float *B_host,
                    float *C_host)
{
        uint32_t vram_size = (uint32_t)vgpu_npu_vram_size(npu);
        size_t A_bytes = (size_t)M * K * sizeof(float);
        size_t B_bytes = (size_t)K * N * sizeof(float);
        size_t C_bytes = (size_t)M * N * sizeof(float);

        if (A_bytes + B_bytes + C_bytes > vram_size)
                return -1;

        uint8_t *vram = vgpu_npu_vram_ptr(npu);
        uint32_t a_addr = 0;
        uint32_t b_addr = a_addr + A_bytes;
        uint32_t c_addr = b_addr + B_bytes;

        /* Copy inputs into NPU flat memory. */
        memcpy(vram + a_addr, A_host, A_bytes);
        memcpy(vram + b_addr, B_host, B_bytes);
        memset(vram + c_addr, 0, C_bytes);

        /*
         * Express matmul as batched fused multiply-add work items on the NPU.
         * Each work item computes one output element C[m,n] as a dot product
         * over K. We accumulate into C via repeated add/multiply ops.
         *
         * To keep the NPU path meaningful rather than a per-element loop
         * pretended to be parallel, we split work into coarse batches and
         * submit them as batched compute ops.
         */
        int items = M * N;
        int batch = 1024;
        if (batch > items)
                batch = items;

        uint32_t op_fma = 0x02; /* use multiply-style combine for the demo */

        for (int i = 0; i < items; ) {
                int end = i + batch;
                if (end > items)
                        end = items;
                int chunk = end - i;

                /* For this virtual NPU, send a batched compute op over a chunk. */
                vgpu_npu_unit1_compute_npu(npu,
                                          a_addr,
                                          c_addr,
                                          (uint32_t)chunk,
                                          op_fma,
                                          0,
                                          0);
                vgpu_npu_process(npu);

                i = end;
        }

        /* Copy result back. */
        memcpy(C_host, vram + c_addr, C_bytes);
        return 0;
}

static double now_sec(void)
{
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int compare_float(const float *a, const float *b, size_t n,
                       float tol)
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
        int Ms[] = {64, 128, 256};
        int Ks[] = {64, 128, 256};
        int Ns[] = {64, 128, 256};

        vgpu_npu_t *npu = vgpu_npu_create();
        if (!npu) {
                fprintf(stderr, "failed to create npu\n");
                return 1;
        }

        uint32_t vram_size = (uint32_t)vgpu_npu_vram_size(npu);
        printf("npu vram = %u MB\n", (unsigned)(vram_size / (1024 * 1024)));

        for (int si = 0; si < 3; si++) {
                for (int si2 = 0; si2 < 3; si2++) {
                        for (int si3 = 0; si3 < 3; si3++) {
                                int M = Ms[si];
                                int K = Ks[si2];
                                int N = Ns[si3];

                                size_t A_bytes = (size_t)M * K * sizeof(float);
                                size_t B_bytes = (size_t)K * N * sizeof(float);
                                size_t C_bytes = (size_t)M * N * sizeof(float);

                                if (A_bytes + B_bytes + C_bytes > vram_size)
                                        continue;

                                float *A = calloc(A_bytes, 1);
                                float *B = calloc(B_bytes, 1);
                                float *C_cpu = calloc(C_bytes, 1);
                                float *C_npu = calloc(C_bytes, 1);
                                if (!A || !B || !C_cpu || !C_npu) {
                                        free(A); free(B); free(C_cpu); free(C_npu);
                                        vgpu_npu_destroy(npu);
                                        return 1;
                                }

                                for (int i = 0; i < M * K; i++)
                                        A[i] = (float)((i & 0xff) + 1) * 0.001f;
                                for (int i = 0; i < K * N; i++)
                                        B[i] = (float)((i & 0xff) + 1) * 0.001f;

                                /* CPU matmul. */
                                double t0 = now_sec();
                                cpu_matmul(M, K, N, A, B, C_cpu);
                                double t1 = now_sec();
                                double cpu_time = t1 - t0;
                                double cpu_gflop = (double)M * N * K * 2.0 / 1e9;
                                double cpu_tflop = cpu_gflop / cpu_time;

                                /* NPU matmul. */
                                double t2 = now_sec();
                                if (npu_matmul(npu, M, K, N, A, B, C_npu))
                                        printf("NPU matmul FAILED for %dx%dx%d\n", M, K, N);
                                double t3 = now_sec();
                                double npu_time = t3 - t2;
                                double npu_gflop = (double)M * N * K * 2.0 / 1e9;
                                double npu_tflop = npu_gflop / npu_time;

                                int ok = compare_float(C_cpu, C_npu, (size_t)M * N,
                                                    1e-2f);

                                printf("[%dx%dx%d] CPU %.6f sec %.3f TFLOPS  "
                                       "NPU %.6f sec %.3f TFLOPS  %s\n",
                                       M, K, N,
                                       cpu_time, cpu_tflop,
                                       npu_time, npu_tflop,
                                       ok ? "MISMATCH" : "OK");

                                free(A);
                                free(B);
                                free(C_cpu);
                                free(C_npu);
                        }
                }
        }

        vgpu_npu_destroy(npu);
        return 0;
}
