#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "virtual_npu.h"

static float *alloc_f(size_t n) { return (float *)calloc(n, sizeof(float)); }

static void cpu_matmul(int M, int K, int N, const float *A, const float *B, float *C)
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
                if (d < 0) d = -d;
                if (d > tol) return 1;
        }
        return 0;
}

int main(void)
{
        int M = 256, K = 256, N = 256;
        printf("Split workload test: %dx%dx%d\n\n", M, K, N);
        
        size_t vram_mb = 512;
        virtual_npu_t *n = virtual_npu_create(vram_mb * 1024 * 1024);
        if (!n) return 1;
        
        float *A = alloc_f(M * K);
        float *B = alloc_f(K * N);
        float *C_cpu = alloc_f(M * N);
        float *C_npu = alloc_f(M * N);
        float *C_split = alloc_f(M * N);
        
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = ((float)(i % 100)) * 0.01f;
        for (size_t i = 0; i < (size_t)K * N; i++) B[i] = ((float)(i % 100)) * 0.01f;
        
        /* Pure CPU */
        double t0 = now_sec();
        cpu_matmul(M, K, N, A, B, C_cpu);
        double t1 = now_sec();
        double cpu_time = t1 - t0;
        printf("CPU only:        %.6f sec  %.2f GFLOPS\n", cpu_time, (double)M*N*K*2.0/1e9/cpu_time);
        
        /* Pure NPU (Voodoo) */
        double t2 = now_sec();
        size_t a_bytes = (size_t)M * K * 4;
        size_t b_bytes = (size_t)K * N * 4;
        size_t c_bytes = (size_t)M * N * 4;
        uint8_t *vram_mem = virtual_npu_memory(n);
        memcpy(vram_mem + 0, A, a_bytes);
        memcpy(vram_mem + a_bytes, B, b_bytes);
        memset(vram_mem + a_bytes + b_bytes, 0, c_bytes);
        virtual_npu_invalidate(n);
        virtual_npu_compute(n, 0, a_bytes, a_bytes + b_bytes, a_bytes + b_bytes + c_bytes,
                           M, K, N, 16, 16, 16, NPU_OP_VOODOO, 0x01, 0);
        virtual_npu_process(n);
        memcpy(C_npu, vram_mem + a_bytes + b_bytes, c_bytes);
        double t3 = now_sec();
        double npu_time = t3 - t2;
        printf("NPU only (Voodoo): %.6f sec  %.2f GFLOPS  %s\n", npu_time, (double)M*N*K*2.0/1e9/npu_time,
               compare_float(C_cpu, C_npu, M*N, 1e-2f) ? "MISMATCH" : "OK");
        
        /* Split 50/50: CPU computes bottom half of rows, NPU computes top half */
        double t4 = now_sec();
        /* CPU computes rows 0 to M/2-1 */
        for (int m = 0; m < M/2; m++) {
                for (int ncol = 0; ncol < N; ncol++) {
                        float acc = 0.0f;
                        for (int k = 0; k < K; k++) {
                                acc += A[m * K + k] * B[k * N + ncol];
                        }
                        C_split[m * N + ncol] = acc;
                }
        }
        /* NPU computes rows M/2 to M-1 */
        memset(vram_mem + a_bytes + b_bytes, 0, c_bytes);
        /* Copy only the A rows the NPU needs */
        memcpy(vram_mem + 0, A + (M/2)*K, (size_t)(M/2)*K*4);
        memcpy(vram_mem + a_bytes, B, b_bytes);
        virtual_npu_invalidate(n);
        virtual_npu_compute(n, 0, a_bytes, a_bytes + b_bytes, a_bytes + b_bytes + c_bytes,
                           M/2, K, N, 16, 16, 16, NPU_OP_VOODOO, 0x01, 0);
        virtual_npu_process(n);
        memcpy(C_split + (M/2)*N, vram_mem + a_bytes + b_bytes, (size_t)(M/2)*N*4);
        double t5 = now_sec();
        double split_time = t5 - t4;
        printf("Split 50/50:     %.6f sec  %.2f GFLOPS  %s\n", split_time, (double)M*N*K*2.0/1e9/split_time,
               compare_float(C_cpu, C_split, M*N, 1e-2f) ? "MISMATCH" : "OK");
        
        printf("\nSpeedup vs CPU: NPU only = %.2fx, Split = %.2fx\n", cpu_time/npu_time, cpu_time/split_time);
        
        virtual_npu_destroy(n);
        free(A); free(B); free(C_cpu); free(C_npu); free(C_split);
        return 0;
}
