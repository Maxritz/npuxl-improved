#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "virtual_npu.h"

static float *alloc_f(size_t n) { return (float *)calloc(n, sizeof(float)); }
static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec/1e9; }

int main() {
    int sizes[] = {64, 128, 192, 256, 384};
    for (int si = 0; si < 5; si++) {
        int M = sizes[si], K = sizes[si], N = sizes[si];
        printf("\n=== %dx%dx%d ===\n", M, K, N);
        virtual_npu_t *n = virtual_npu_create(1024*1024*1024);
        float *A = alloc_f(M*K), *B = alloc_f(K*N), *C = alloc_f(M*N);
        for (int i = 0; i < M*K; i++) A[i] = (float)(i%100)*0.001f;
        for (int i = 0; i < K*N; i++) B[i] = (float)(i%100)*0.001f;
        size_t a_bytes = M*K*4, b_bytes = K*N*4, c_bytes = M*N*4;
        uint8_t *vram = virtual_npu_memory(n);
        memcpy(vram, A, a_bytes); memcpy(vram+a_bytes, B, b_bytes);
        
        memset(vram+a_bytes+b_bytes, 0, c_bytes);
        virtual_npu_invalidate(n);
        double t0 = now_sec();
        virtual_npu_compute(n, 0, a_bytes, a_bytes+b_bytes, a_bytes+b_bytes+c_bytes,
                           M,K,N, 16,16,16, NPU_OP_MERGED, 0x01, 0);
        virtual_npu_process(n);
        double t1 = now_sec();
        printf("MERGED: %.4f ms (%.1f GFLOPS)\n", (t1-t0)*1000, (double)M*N*K*2.0/1e9/(t1-t0));
        
        memset(vram+a_bytes+b_bytes, 0, c_bytes);
        virtual_npu_invalidate(n);
        t0 = now_sec();
        virtual_npu_compute(n, 0, a_bytes, a_bytes+b_bytes, a_bytes+b_bytes+c_bytes,
                           M,K,N, 16,16,16, NPU_OP_VOODOO, 0x01, 0);
        virtual_npu_process(n);
        t1 = now_sec();
        printf("VOODOO: %.4f ms (%.1f GFLOPS)\n", (t1-t0)*1000, (double)M*N*K*2.0/1e9/(t1-t0));
        
        virtual_npu_destroy(n);
        free(A); free(B); free(C);
    }
    return 0;
}
