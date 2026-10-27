#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "virtual_npu.h"

static void cpu_matmul_2x2(
    const float A[4],
    const float B[4],
    float C[4])
{
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            C[r * 2 + c] =
                A[r * 2]     * B[c] +
                A[r * 2 + 1] * B[2 + c];
        }
    }
}

int main(void)
{
    const float A[4] = {
        1, 2,
        3, 4
    };

    const float B[4] = {
        5, 6,
        7, 8
    };

    float C_cpu[4] = {0};
    cpu_matmul_2x2(A, B, C_cpu);

    printf("CPU reference:\n");
    for (int i = 0; i < 4; ++i)
        printf("C[%d] = %.6f\n", i, C_cpu[i]);
    printf("\n");

    virtual_npu_t *n = virtual_npu_create(1024 * 1024);
    if (!n) {
        fprintf(stderr, "failed to create npu\n");
        return 1;
    }

    uint8_t *vram = virtual_npu_memory(n);
    size_t a_addr = 0;
    size_t b_addr = 4 * sizeof(float);
    size_t c_addr = 8 * sizeof(float);
    size_t ctl_addr = 12 * sizeof(float);

    memcpy(vram + a_addr, A, 4 * sizeof(float));
    memcpy(vram + b_addr, B, 4 * sizeof(float));
    memset(vram + c_addr, 0, 4 * sizeof(float));

    /* Test Voodoo path */
    virtual_npu_invalidate(n);
    virtual_npu_compute(n, a_addr, b_addr, c_addr, ctl_addr, 2, 2, 2, 2, 2, 2, NPU_OP_VOODOO, 0x01, 0);
    virtual_npu_process(n);

    float C_voodoo[4];
    memcpy(C_voodoo, vram + c_addr, 4 * sizeof(float));

    printf("Voodoo path:\n");
    for (int i = 0; i < 4; ++i)
        printf("C[%d] = %.6f\n", i, C_voodoo[i]);
    printf("Match: %s\n\n", (fabsf(C_voodoo[0]-19.0f)<1e-5 && fabsf(C_voodoo[1]-22.0f)<1e-5 && fabsf(C_voodoo[2]-43.0f)<1e-5 && fabsf(C_voodoo[3]-50.0f)<1e-5) ? "YES" : "NO");

    /* Test Wildcat path */
    virtual_npu_invalidate(n);
    virtual_npu_compute(n, a_addr, b_addr, c_addr, ctl_addr, 2, 2, 2, 2, 2, 2, NPU_OP_WILDCAT, 0x01, 0);
    virtual_npu_process(n);

    float C_wildcat[4];
    memcpy(C_wildcat, vram + c_addr, 4 * sizeof(float));

    printf("Wildcat path:\n");
    for (int i = 0; i < 4; ++i)
        printf("C[%d] = %.6f\n", i, C_wildcat[i]);
    printf("Match: %s\n\n", (fabsf(C_wildcat[0]-19.0f)<1e-5 && fabsf(C_wildcat[1]-22.0f)<1e-5 && fabsf(C_wildcat[2]-43.0f)<1e-5 && fabsf(C_wildcat[3]-50.0f)<1e-5) ? "YES" : "NO");

    /* Test CPU path */
    virtual_npu_invalidate(n);
    virtual_npu_compute(n, a_addr, b_addr, c_addr, ctl_addr, 2, 2, 2, 2, 2, 2, NPU_OP_CPU, 0x01, 0);
    virtual_npu_process(n);

    float C_ncpu[4];
    memcpy(C_ncpu, vram + c_addr, 4 * sizeof(float));

    printf("NPU CPU path:\n");
    for (int i = 0; i < 4; ++i)
        printf("C[%d] = %.6f\n", i, C_ncpu[i]);
    printf("Match: %s\n", (fabsf(C_ncpu[0]-19.0f)<1e-5 && fabsf(C_ncpu[1]-22.0f)<1e-5 && fabsf(C_ncpu[2]-43.0f)<1e-5 && fabsf(C_ncpu[3]-50.0f)<1e-5) ? "YES" : "NO");

    virtual_npu_destroy(n);
    return 0;
}
