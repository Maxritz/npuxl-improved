#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "virtual_npu.h"

static float *alloc_f(size_t n) { return (float *)calloc(n, sizeof(float)); }

int main() {
    printf("Sherlock-IT tracing enabled\n");
    sherlock_enable_tracing(1);
    
    int M = 256, K = 256, N = 256;
    virtual_npu_t *n = virtual_npu_create(1024*1024*1024);
    float *A = alloc_f(M*K), *B = alloc_f(K*N), *C = alloc_f(M*N);
    for (int i = 0; i < M*K; i++) A[i] = (float)(i%100)*0.001f;
    for (int i = 0; i < K*N; i++) B[i] = (float)(i%100)*0.001f;
    
    size_t a_bytes = M*K*4, b_bytes = K*N*4, c_bytes = M*N*4;
    uint8_t *vram = virtual_npu_memory(n);
    memcpy(vram, A, a_bytes); memcpy(vram+a_bytes, B, b_bytes);
    memset(vram+a_bytes+b_bytes, 0, c_bytes);
    
    sherlock_reset_trace();
    virtual_npu_invalidate(n);
    virtual_npu_compute(n, 0, a_bytes, a_bytes+b_bytes, a_bytes+b_bytes+c_bytes,
                       M,K,N, 32,32,32, NPU_OP_MERGED, 0x01, 0);
    virtual_npu_process(n);
    sherlock_dump_trace("MERGED-256x256x256");
    
    memset(vram+a_bytes+b_bytes, 0, c_bytes);
    sherlock_reset_trace();
    virtual_npu_invalidate(n);
    virtual_npu_compute(n, 0, a_bytes, a_bytes+b_bytes, a_bytes+b_bytes+c_bytes,
                       M,K,N, 32,32,32, NPU_OP_VOODOO, 0x01, 0);
    virtual_npu_process(n);
    sherlock_dump_trace("VOODOO-256x256x256");
    
    virtual_npu_destroy(n);
    free(A); free(B); free(C);
    sherlock_enable_tracing(0);
    return 0;
}
