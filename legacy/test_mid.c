#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "virtual_npu.h"

static float *alloc_f(size_t n) { return (float *)calloc(n, sizeof(float)); }
static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec/1e9; }

static void cpu_matmul_ref(int M, int K, int N, const float *A, const float *B, float *C)
{
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += A[m*K+k]*B[k*N+n];
            C[m*N+n] = acc;
        }
    }
}

int main() {
    int M=1024, K=1024, N=1024;
    printf("Large dataset trace: %dx%dx%d (%.2f GFLOP workload)\n", M,K,N, (double)M*N*K*2.0/1e9);
    printf("===============================================\n\n");
    
    sherlock_enable_tracing(1);
    
    float *A = alloc_f(M*K), *B = alloc_f(K*N), *C_cpu = alloc_f(M*N);
    for (int i=0;i<M*K;i++) A[i]=(float)(rand()%100)*0.01f;
    for (int i=0;i<K*N;i++) B[i]=(float)(rand()%100)*0.01f;
    
    printf("[CPU BASELINE]\n");
    double t0 = now_sec();
    cpu_matmul_ref(M,K,N,A,B,C_cpu);
    double t1 = now_sec();
    double cpu_time = t1-t0;
    printf("Time: %.4f ms, Performance: %.2f GFLOPS\n\n", cpu_time*1000, (double)M*N*K*2.0/1e9/cpu_time);
    
    virtual_npu_t *n = virtual_npu_create(4ULL*1024*1024*1024);
    size_t a_bytes=M*K*4, b_bytes=K*N*4, c_bytes=M*N*4;
    uint8_t *vram=virtual_npu_memory(n);
    memcpy(vram,A,a_bytes); memcpy(vram+a_bytes,B,b_bytes);
    
    struct {const char *name; int op;} tests[] = {
        {"MERGED (Fused Tiled)", NPU_OP_MERGED},
        {"TENSOR_CORE (16x16x16)", NPU_OP_TENSOR_CORE},
    };
    
    for (int t=0;t<2;t++) {
        printf("[%s]\n", tests[t].name);
        memset(vram+a_bytes+b_bytes,0,c_bytes);
        sherlock_reset_trace();
        virtual_npu_invalidate(n);
        double t2 = now_sec();
        virtual_npu_compute(n,0,a_bytes,a_bytes+b_bytes,a_bytes+b_bytes+c_bytes,M,K,N,128,128,128,tests[t].op,0x01,0);
        virtual_npu_process(n);
        double t3 = now_sec();
        double npu_time = t3-t2;
        double speedup = cpu_time / npu_time;
        printf("Time: %.4f ms, Performance: %.2f GFLOPS, Speedup vs CPU: %.2fx\n", 
               npu_time*1000, (double)M*N*K*2.0/1e9/npu_time, speedup);
        sherlock_dump_trace(tests[t].name);
    }
    
    virtual_npu_destroy(n);
    free(A);free(B);free(C_cpu);
    sherlock_enable_tracing(0);
    printf("\n===============================================\n");
    printf("Sherlock-IT Analysis Complete\n");
    return 0;
}
