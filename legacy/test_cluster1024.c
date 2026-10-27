#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
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

typedef struct {
    virtual_npu_t *npu;
    int start_row, end_row;
    int M,K,N;
    float *A, *B, *C;
} thread_arg_t;

static void *npu_worker(void *arg)
{
    thread_arg_t *t = (thread_arg_t *)arg;
    int rows = t->end_row - t->start_row;
    if (rows <= 0) return NULL;
    size_t a_bytes = rows * t->K * 4;
    size_t b_bytes = t->K * t->N * 4;
    size_t c_bytes = rows * t->N * 4;
    uint8_t *vram = virtual_npu_memory(t->npu);
    memcpy(vram, t->A + t->start_row * t->K, a_bytes);
    memcpy(vram + a_bytes, t->B, b_bytes);
    memset(vram + a_bytes + b_bytes, 0, c_bytes);
    virtual_npu_invalidate(t->npu);
    virtual_npu_compute(t->npu, 0, a_bytes, a_bytes + b_bytes, 0, rows, t->K, t->N, 128, 128, 128, NPU_OP_MERGED, 0x01, 0);
    virtual_npu_process(t->npu);
    memcpy(t->C + t->start_row * t->N, vram + a_bytes + b_bytes, c_bytes);
    return NULL;
}

int main() {
    int M=1024, K=1024, N=1024;
    printf("2-NPU Cluster vs CPU: %dx%dx%d\n", M,K,N);
    printf("===============================================\n\n");
    
    float *A = alloc_f(M*K), *B = alloc_f(K*N), *C_cpu = alloc_f(M*N), *C_cluster = alloc_f(M*N);
    for (int i=0;i<M*K;i++) A[i]=(float)(rand()%100)*0.01f;
    for (int i=0;i<K*N;i++) B[i]=(float)(rand()%100)*0.01f;
    
    printf("[CPU BASELINE]\n");
    double t0 = now_sec();
    cpu_matmul_ref(M,K,N,A,B,C_cpu);
    double t1 = now_sec();
    double cpu_time = t1-t0;
    printf("Time: %.4f ms, Performance: %.2f GFLOPS\n\n", cpu_time*1000, (double)M*N*K*2.0/1e9/cpu_time);
    
    printf("[2-NPU CLUSTER (parallel)]\n");
    virtual_npu_t *npu1 = virtual_npu_create(4ULL*1024*1024*1024);
    virtual_npu_t *npu2 = virtual_npu_create(4ULL*1024*1024*1024);
    pthread_t pt1,pt2;
    thread_arg_t arg1 = {npu1, 0, M/2, M, K, N, A, B, C_cluster};
    thread_arg_t arg2 = {npu2, M/2, M, M, K, N, A, B, C_cluster};
    double ts = now_sec();
    pthread_create(&pt1, NULL, npu_worker, &arg1);
    pthread_create(&pt2, NULL, npu_worker, &arg2);
    pthread_join(pt1, NULL);
    pthread_join(pt2, NULL);
    double te = now_sec();
    double cluster_time = te - ts;
    printf("Time: %.4f ms, Performance: %.2f GFLOPS, Speedup vs CPU: %.2fx\n\n", 
           cluster_time*1000, (double)M*N*K*2.0/1e9/cluster_time, cpu_time/cluster_time);
    
    virtual_npu_destroy(npu1);
    virtual_npu_destroy(npu2);
    free(A);free(B);free(C_cpu);free(C_cluster);
    return 0;
}
