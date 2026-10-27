/*
 * Benchmark: CPU vs Voodoo-instruction vs Wildcat-instruction compute.
 *
 * This sends the same typed compute job to the virtual NPU three ways and
 * compares each against a direct CPU implementation.
 *
 * Job:
 *   For each work item:
 *     r = a*b + c*d
 *   with an optional cull filter that skips some items.
 *
 * We run each style:
 *   - CPU direct
 *   - NPU with NPU_OP_VOODOO
 *   - NPU with NPU_OP_WILDCAT
 *
 * Then compare correctness and report time/throughput/instructions.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "virtual_npu.h"

static double now_sec(void)
{
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static void cpu_compute_jjob(uint8_t *base, size_t count)
{
        for (size_t i = 0; i < count; i++) {
                uint8_t *base_i = base + i * 16;
                uint32_t a, b, c, d;
                memcpy(&a, base_i + 0, 4);
                memcpy(&b, base_i + 4, 4);
                memcpy(&c, base_i + 8, 4);
                memcpy(&d, base_i + 12, 4);
                float af = (float)a, bf = (float)b, cf = (float)c, df = (float)d;
                float r = af * bf + cf * df;
                uint32_t rv;
                memcpy(&rv, &r, 4);
                *(uint32_t *)(base_i + 12) = rv;
        }
}

static int run_and_compare(const char *label,
                        virtual_npu_t *n,
                        uint8_t *buf,
                        size_t count,
                        uint32_t op_code,
                        uint32_t op_kind,
                        uint32_t cull_threshold,
                        int cull_enable)
{
        uint8_t *ref = (uint8_t *)malloc(count * 16);
        if (!ref)
                return -1;
        memcpy(ref, buf, count * 16);

        double t0 = now_sec();
        cpu_compute_jjob(ref, count);
        double t1 = now_sec();
        double cpu_time = t1 - t0;

        memcpy(buf, ref, count * 16);
        if (cull_enable) {
                for (size_t i = 0; i < count; i++) {
                        uint32_t ctl = ((uint32_t *)buf)[i] & 0xffffffff;
                        if (ctl < cull_threshold)
                                ((uint32_t *)buf)[i] = 0;
                }
        }

        virtual_npu_compute(n,
                            0,
                            count,
                            op_kind,
                            op_code,
                            cull_enable ? 0 : 0,
                            cull_threshold);

        virtual_npu_process(n);

        double t2 = now_sec();
        double npu_time = t2 - t0;

        virtual_npu_status_t s;
        virtual_npu_read_status(n, &s);

        int ok = 1;
        for (size_t i = 0; i < count; i++) {
                uint32_t cpu_r, npu_r;
                memcpy(&cpu_r, ref + i * 16 + 12, 4);
                memcpy(&npu_r, buf + i * 16 + 12, 4);
                if (cpu_r != npu_r) {
                        if (ok) {
                                printf("  first mismatch @%zu cpu=%u npu=%u\n",
                                       i, cpu_r, npu_r);
                                ok = 0;
                        }
                }
        }

        printf("%-10s count=%-10zu op=0x%02x cull=%d  "
               "time=%.6f sec  ops=%llu  processed=%llu  culled=%llu "
               "instructions=%llu  match=%s\n",
               label,
               count,
               op_code,
               cull_enable ? cull_threshold : 0,
               npu_time,
               (unsigned long long)s.ops,
               (unsigned long long)s.processed,
               (unsigned long long)s.culled,
               (unsigned long long)s.instructions,
               ok ? "OK" : "MISMATCH");

        free(ref);
        return ok ? 0 : 1;
}

int main(void)
{
        size_t vram_mb = 256;
        virtual_npu_t *n = virtual_npu_create(vram_mb * 1024 * 1024);
        if (!n) {
                fprintf(stderr, "failed to create virtual npu\n");
                return 1;
        }

        printf("virtual npu vram = %zu MB\n\n", vram_mb);

        size_t counts[] = {1 << 16, 1 << 18, 1 << 20};
        uint32_t ops[] = {0x01, 0x02, 0x03};

        for (int ci = 0; ci < 3; ci++) {
                size_t count = counts[ci];
                uint32_t cull_threshold = 1;

                uint8_t *buf = (uint8_t *)malloc(count * 16);
                if (!buf) {
                        virtual_npu_destroy(n);
                        return 1;
                }

                for (int i = 0; i < (int)count; i++) {
                        uint8_t *base = buf + i * 16;
                        uint32_t a = (uint32_t)(i & 0xffff) + 1;
                        uint32_t b = (uint32_t)((i >> 8) & 0xffff) + 1;
                        uint32_t c = (uint32_t)((i >> 16) & 0xffff) + 1;
                        uint32_t d = (uint32_t)((i >> 24) & 0xffff) + 1;
                        memcpy(base + 0, &a, 4);
                        memcpy(base + 4, &b, 4);
                        memcpy(base + 8, &c, 4);
                        memcpy(base + 12, &d, 4);
                }

                printf("=== count=%zu cull_threshold=%u ===\n", count, cull_threshold);

                int failures = 0;

                failures += run_and_compare("CPU", n, buf, count, ops[ci], NPU_OP_CPU, cull_threshold, 0);
                failures += run_and_compare("Voodoo", n, buf, count, ops[ci], NPU_OP_VOODOO, cull_threshold, 1);
                failures += run_and_compare("Wildcat", n, buf, count, ops[ci], NPU_OP_WILDCAT, cull_threshold, 1);

                printf("\n");
                free(buf);
        }

        virtual_npu_destroy(n);
        return 0;
}
