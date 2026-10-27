#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "virtual_npu.h"

static float *alloc_f(size_t n) { return (float *)calloc(n, sizeof(float)); }
static double now_sec(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec/1e9; }

int main() {
    int M=256,K=256,N=256;
    printf("All kernels test %dx%dx%d\n\n", M,K,N);
    virtual_npu_t *n = virtual_npu_create(1024*1024*1024);
    float *A = alloc_f(M*K), *B = alloc_f(K*N), *C = alloc_f(M*N);
    for (int i=0;i<M*K;i++) A[i]=(float)(i%100)*0.001f;
    for (int i=0;i<K*N;i++) B[i]=(float)(i%100)*0.001f;
    size_t a=M*K*4,b=K*N*4,c=M*N*4;
    uint8_t *vram=virtual_npu_memory(n);
    memcpy(vram,A,a); memcpy(vram+a,B,b);
    
    struct {const char *name; int op;} tests[] = {
        {"CPU", NPU_OP_CPU},
        {"VOODOO", NPU_OP_VOODOO},
        {"WILDCAT", NPU_OP_WILDCAT},
        {"MERGED", NPU_OP_MERGED},
        {"SHADER_GEMM", NPU_OP_SHADER_GEMM},
        {"TENSOR_CORE", NPU_OP_TENSOR_CORE},
        {"FUSED_GEMM", NPU_OP_FUSED_GEMM},
    };
    for (int t=0;t<7;t++) {
        memset(vram+a+b,0,c);
        virtual_npu_invalidate(n);
        double t0=now_sec();
        virtual_npu_compute(n,0,a,a+b,a+b+c,M,K,N,32,32,32,tests[t].op,0x01,0);
        virtual_npu_process(n);
        double t1=now_sec();
        printf("%s: %.4f ms (%.1f GFLOPS)\n", tests[t].name, (t1-t0)*1000, (double)M*N*K*2.0/1e9/(t1-t0));
    }
    virtual_npu_destroy(n);
    free(A);free(B);free(C);
    return 0;
}
