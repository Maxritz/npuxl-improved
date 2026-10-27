/*
 * Virtual NPU batched/tensor matmul benchmark: CPU vs Voodoo vs Wildcat.
 *
 * This sends C = A * B matmul to the virtual NPU three ways and compares each
 * against a direct CPU reference.
 *
 * NPU memory layout:
 *   A at a_addr,  shape [M, K], size M*K elements
 *   B at b_addr,  shape [K, N], size K*N elements
 *   C at c_addr,  shape [M, N], size M*N elements, output
 *   ctl at ctl_addr, shape [M, N], size M*N elements, optional cull control
 *
 * Compute styles:
 *   CPU     - reference tiled matmul
 *   Voodoo  - batched command-stream tile dispatch
 *   Wildcat - programmable compute unit style with optional tile culling
 *
 * Tiles use triangle clamping at matrix edges, so partial tiles are handled
 * correctly.
 *
 * Output:
 *   - per-shape CPU time + TFLOPS
 *   - per-shape NPU CPU time + TFLOPS + dispatches/batches/ops/processed/culled
 *   - per-shape NPU Voodoo time + TFLOPS + dispatches/batches/ops/processed/culled
 *   - per-shape NPU Wildcat time + TFLOPS + dispatches/batches/ops/processed/culled
 *   - correctness vs CPU
 *
 * The Relumination lights the scene as it goes, so shading is no longer a
 * separate footstep pass. Railroad switch state is read once up front and
 * written back once at the end. This is a deliberate run-to-completion step
 * toward a framed, lit simulation that still uses 2 shader passes.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "virtual_npu.h"

static float *alloc_f(size_t n)
{
        return (float *)calloc(n, sizeof(float));
}

static float *safe_free(float *p)
{
        free(p);
        return NULL;
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

/*
 * Put a matmul job into the NPU flat memory and run one compute style.
 *
 * Layout in NPU memory:
 *   a_addr,  size M*K
 *   b_addr,  size K*N
 *   c_addr,  size M*N, initialized to 0
 *   ctl_addr, size M*N, optional cull control
 */
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
                        printf("  first mismatch @%zu cpu=%g npu=%g\n", i, a[i], b[i]);
                        return 1;
                }
        }
        return 0;
}

int main(void)
{
        int shapes[][4] = {
                /* M, K, N, cull_enable */
                {32, 32, 32, 0},
                {64, 64, 64, 0},
                {96, 96, 96, 0},
                {128, 128, 128, 0},
                {128, 128, 128, 1},
        };
        int nshapes = sizeof(shapes) / sizeof(shapes[0]);

        size_t vram_mb = 512;
        virtual_npu_t *n = virtual_npu_create(vram_mb * 1024 * 1024);
        if (!n) {
                fprintf(stderr, "failed to create virtual npu\n");
                return 1;
        }

        printf("virtual npu vram = %zu MB\n\n", vram_mb);

        double cpu_total = 0.0;
        double voodoo_total = 0.0;
        double wildcat_total = 0.0;
        uint64_t cpu_ops_total = 0;
        uint64_t voodoo_ops_total = 0;
        uint64_t wildcat_ops_total = 0;

        for (int s = 0; s < nshapes; s++) {
                int M = shapes[s][0];
                int K = shapes[s][1];
                int N = shapes[s][2];
                int cull_enable = shapes[s][3];

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
                        A[i] = (float)((i & 0xff) + 1) * 0.001f;
                for (size_t i = 0; i < (size_t)K * N; i++)
                        B[i] = (float)((i & 0xff) + 1) * 0.001f;

                {
                        double t0 = now_sec();
                        cpu_matmul(M, K, N, A, B, C_cpu);
                        double t1 = now_sec();
                        double cpu_time = t1 - t0;
                        double cpu_gflop = (double)M * N * K * 2.0 / 1e9;
                        double cpu_tflop = cpu_gflop / cpu_time;
                        cpu_total += cpu_time;
                        cpu_ops_total += (uint64_t)M * N * K * 2;

                        printf("[%dx%dx%d] CPU  %8.6f sec  %8.3f TFLOPS\n",
                               M, K, N, cpu_time, cpu_tflop);
                }

                {
                        size_t tile_m = 4;
                        size_t tile_k = 8;
                        size_t tile_n = 4;

                        virtual_npu_invalidate(n);

                        double t2 = now_sec();
                        if (npu_matmul(n, M, K, N, A, B, C_cpu2,
                                        tile_m, tile_k, tile_n,
                                        NPU_OP_CPU, 0x01, cull_enable ? 1 : 0)) {
                                printf("  NPU CPU FAILED\n");
                                virtual_npu_dump(n);
                                continue;
                        }
                        double t3 = now_sec();
                        double npu_cpu_time = t3 - t2;
                        double npu_cpu_gflop = (double)M * N * K * 2.0 / 1e9;
                        double npu_cpu_tflop = npu_cpu_gflop / npu_cpu_time;

                        virtual_npu_status_t vs;
                        virtual_npu_read_status(n, &vs);
                        printf(
                                "        NPU CPU %8.6f sec  %8.3f TFLOPS  "
                                "disp=%llu bat=%llu ops=%llu proc=%llu cull=%llu  %s\n",
                                npu_cpu_time, npu_cpu_tflop,
                                (unsigned long long)vs.dispatches,
                                (unsigned long long)vs.batches,
                                (unsigned long long)vs.ops,
                                (unsigned long long)vs.processed,
                                (unsigned long long)vs.culled,
                                compare_float(C_cpu, C_cpu2, (size_t)M * N, 1e-2f)
                                        ? "MISMATCH"
                                        : "OK");
                }

                {
                        size_t tile_m = 4;
                        size_t tile_k = 8;
                        size_t tile_n = 4;

                        virtual_npu_invalidate(n);

                        double t2 = now_sec();
                        if (npu_matmul(n, M, K, N, A, B, C_voodoo,
                                        tile_m, tile_k, tile_n,
                                        NPU_OP_VOODOO, 0x01, cull_enable ? 1 : 0)) {
                                printf("  Voodoo FAILED\n");
                                virtual_npu_dump(n);
                                continue;
                        }
                        double t3 = now_sec();
                        double voodoo_time = t3 - t2;
                        double voodoo_gflop = (double)M * N * K * 2.0 / 1e9;
                        double voodoo_tflop = voodoo_gflop / voodoo_time;
                        voodoo_total += voodoo_time;

                        virtual_npu_status_t vs;
                        virtual_npu_read_status(n, &vs);
                        voodoo_ops_total = vs.ops;

                        printf(
                                "        Voodoo %8.6f sec  %8.3f TFLOPS  "
                                "disp=%llu bat=%llu ops=%llu proc=%llu cull=%llu  %s\n",
                                voodoo_time, voodoo_tflop,
                                (unsigned long long)vs.dispatches,
                                (unsigned long long)vs.batches,
                                (unsigned long long)vs.ops,
                                (unsigned long long)vs.processed,
                                (unsigned long long)vs.culled,
                                compare_float(C_cpu, C_voodoo, (size_t)M * N, 1e-2f)
                                        ? "MISMATCH"
                                        : "OK");
                }

                {
                        size_t tile_m = 4;
                        size_t tile_k = 8;
                        size_t tile_n = 4;

                        virtual_npu_invalidate(n);

                        double t4 = now_sec();
                        if (npu_matmul(n, M, K, N, A, B, C_wildcat,
                                        tile_m, tile_k, tile_n,
                                        NPU_OP_WILDCAT, 0x01, cull_enable ? 1 : 0)) {
                                printf("  Wildcat FAILED\n");
                                virtual_npu_dump(n);
                                continue;
                        }
                        double t5 = now_sec();
                        double wildcat_time = t5 - t4;
                        double wildcat_gflop = (double)M * N * K * 2.0 / 1e9;
                        double wildcat_tflop = wildcat_gflop / wildcat_time;
                        wildcat_total += wildcat_time;

                        virtual_npu_status_t vs;
                        virtual_npu_read_status(n, &vs);
                        wildcat_ops_total = vs.ops;

                        printf(
                                "        Wildcat %8.6f sec  %8.3f TFLOPS  "
                                "disp=%llu bat=%llu ops=%llu proc=%llu cull=%llu  %s\n",
                                wildcat_time, wildcat_tflop,
                                (unsigned long long)vs.dispatches,
                                (unsigned long long)vs.batches,
                                (unsigned long long)vs.ops,
                                (unsigned long long)vs.processed,
                                (unsigned long long)vs.culled,
                                compare_float(C_cpu, C_wildcat, (size_t)M * N, 1e-2f)
                                        ? "MISMATCH"
                                        : "OK");
                }

                printf("\n");

                A = safe_free(A);
                B = safe_free(B);
                C_cpu = safe_free(C_cpu);
                C_cpu2 = safe_free(C_cpu2);
                C_voodoo = safe_free(C_voodoo);
                C_wildcat = safe_free(C_wildcat);
        }

        printf("summary (all shapes):\n");
        printf("  CPU     total %8.6f sec\n", cpu_total);
        printf("  Voodoo  total %8.6f sec\n", voodoo_total);
        printf("  Wildcat total %8.6f sec\n", wildcat_total);
        printf("  CPU     ops   %llu\n", (unsigned long long)cpu_ops_total);
        printf("  Voodoo  ops   %llu\n", (unsigned long long)voodoo_ops_total);
        printf("  Wildcat ops   %llu\n", (unsigned long long)wildcat_ops_total);

        virtual_npu_destroy(n);
        return 0;
}
