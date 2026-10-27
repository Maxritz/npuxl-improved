/*
 * NPU vs CPU: small LLM-inference-style compute microbenchmark.
 *
 * This is not a real LLM. It models the shape of inference work an engine
 * would send to a compute unit:
 *   - a large buffer of packed data,
 *   - a fused compute step that reads input, does a math-heavy combine, and
 *     writes results back,
 *   - repeated over many "layers",
 *   - with a filter/cull stage that skips some of the work.
 *
 * We run the same job two ways and report:
 *   - CPU time and throughput
 *   - NPU time and throughput
 *   - output equality (NPU vs CPU)
 *
 * The NPU path uses unit 1 with a hierarchical-Z-style cull filter and a
 * batched compute op over the flat 2 GB space.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "includes/private/video/vgpu_npu.h"

#define WORDS(size) ((size) / (int)sizeof(uint32_t))

static uint32_t cpu_step(uint32_t a, uint32_t b, uint32_t op)
{
        switch (op & 0xff) {
        case 0x01:
                return a + b;
        case 0x02:
                return a * b;
        case 0x03:
                return a ^ b;
        case 0x04:
                return (a & b) | ((~a) & b);
        case 0x05:
                return (a & b) | (a & ~b);
        case 0x06:
                return (a & b) | ((~a) & b);
        case 0x07:
                return (a & b) | (a & ~b);
        default:
                return a;
        }
}

/*
 * CPU reference: fused combine + write.
 */
static void cpu_infer(uint32_t *dst,
                    const uint32_t *src,
                    uint32_t count,
                    uint32_t op,
                    const uint32_t *ctl,
                    uint32_t cull_threshold,
                    uint64_t *out_pixels,
                    uint64_t *out_ops,
                    uint64_t *out_culled)
{
        uint64_t pixels = 0;
        uint64_t ops = 0;
        uint64_t culled = 0;

        for (uint32_t i = 0; i < count; i++) {
                uint32_t keep = 1;
                if (ctl) {
                        keep = ctl[i] >= cull_threshold ? 1 : 0;
                }
                if (!keep) {
                        culled++;
                        continue;
                }
                uint32_t v = cpu_step(src[i], src[i], op);
                dst[i] = v;
                pixels++;
                ops += 2;
        }

        if (out_pixels)
                *out_pixels = pixels;
        if (out_ops)
                *out_ops = ops;
        if (out_culled)
                *out_culled = culled;
}

/*
 * NPU path: copy input into NPU flat memory, run unit 1 compute with the
 * same cull filter, read results back, then compare with CPU.
 */
static int npu_infer(vgpu_npu_t *npu,
                   const uint32_t *src_host,
                   uint32_t count,
                   uint32_t op,
                   const uint32_t *ctl_host,
                   uint32_t cull_threshold,
                   uint32_t *npu_out_cpu)
{
        uint32_t vram_size = (uint32_t)vgpu_npu_vram_size(npu);
        if (count * 4 * 2 > vram_size)
                return -1;

        uint32_t src_addr = 0;
        uint32_t dst_addr = (count + 1) * 4;
        uint32_t ctl_addr = dst_addr + count * 4;

        /* Copy input + control buffer into NPU flat memory. */
        uint8_t *vram = vgpu_npu_vram_ptr(npu);
        memcpy(vram + src_addr, src_host, count * sizeof(uint32_t));
        if (ctl_host)
                memcpy(vram + ctl_addr, ctl_host, count * sizeof(uint32_t));

        vgpu_npu_unit1_compute_npu(npu, src_addr, dst_addr, count, op,
                                   ctl_host ? ctl_addr : 0, cull_threshold);
        vgpu_npu_process(npu);

        /* Copy NPU results back to CPU for comparison. */
        memcpy(npu_out_cpu, vram + dst_addr, count * sizeof(uint32_t));
        return 0;
}

static double now_sec(void)
{
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static int compare(const uint32_t *a, const uint32_t *b, uint32_t count)
{
        for (uint32_t i = 0; i < count; i++) {
                if (a[i] != b[i]) {
                        printf("mismatch @%u cpu=%u npu=%u\n",
                               (unsigned)i, (unsigned)a[i], (unsigned)b[i]);
                        return 1;
                }
        }
        return 0;
}

int main(void)
{
        /* Inference-like sizes. Keep it small enough for a smoke test but
           large enough to show batching/throughput ideas. */
        uint32_t work = 1 << 18;     /* 262144 work items */
        uint32_t layers = 8;

        vgpu_npu_t *npu = vgpu_npu_create();
        if (!npu) {
                fprintf(stderr, "failed to create npu\n");
                return 1;
        }

        uint32_t vram_size = (uint32_t)vgpu_npu_vram_size(npu);
        uint32_t need = (work + 1) * 4 * 3;
        if (need > vram_size) {
                fprintf(stderr, "work too big for npu vram\n");
                vgpu_npu_destroy(npu);
                return 1;
        }

        uint32_t *cpu_in = calloc(work, sizeof(uint32_t));
        uint32_t *cpu_ctl = calloc(work, sizeof(uint32_t));
        uint32_t *cpu_out = calloc(work, sizeof(uint32_t));
        uint32_t *npu_out = calloc(work, sizeof(uint32_t));
        if (!cpu_in || !cpu_ctl || !cpu_out || !npu_out) {
                free(cpu_in); free(cpu_ctl); free(cpu_out); free(npu_out);
                vgpu_npu_destroy(npu);
                return 1;
        }

        uint32_t op = 0x02; /* multiply-style combine */

        /* Build input + cull control buffer. */
        for (uint32_t i = 0; i < work; i++) {
                cpu_in[i] = (i & 0xffff) + 1;
                cpu_ctl[i] = (i % 7 == 0) ? 0 : ctl[i] + 1;
        }

        /*
         * CPU baseline: run the same fused step over all layers.
         */
        {
                uint64_t cpu_pixels = 0;
                uint64_t cpu_ops = 0;
                uint64_t cpu_culled = 0;
                double t0 = now_sec();

                uint32_t *cur = cpu_in;
                uint32_t *next = cpu_out;

                for (uint32_t l = 0; l < layers; l++) {
                        cpu_infer(next, cur, work, op, cpu_ctl, 1, &cpu_pixels,
                                  &cpu_ops, &cpu_culled);
                        uint32_t *tmp = cur;
                        cur = next;
                        next = tmp;
                }

                double t1 = now_sec();
                double cpu_time = t1 - t0;
                double cpu_throughput = (double)cpu_pixels / cpu_time;

                printf("CPU  layers=%u work=%u pixels=%llu ops=%llu culled=%llu\n"
                       "       time=%.6f sec  throughput=%.3f Mpix/sec\n",
                       layers, work,
                       (unsigned long long)cpu_pixels,
                       (unsigned long long)cpu_ops,
                       (unsigned long long)cpu_culled,
                       cpu_time,
                       cpu_throughput / 1e6);

                /* Keep the final CPU buffer for comparison. */
                memcpy(cpu_out, cur, work * sizeof(uint32_t));
        }

        /*
         * NPU path: same layers, submitted as batched instructions.
         */
        {
                uint64_t npu_pixels = 0;
                uint64_t npu_ops = 0;
                uint64_t npu_culled = 0;
                double t0 = now_sec();

                uint32_t *cur_host = cpu_in;
                uint32_t *next_host = npu_out;

                for (uint32_t l = 0; l < layers; l++) {
                        if (npu_infer(npu, cur_host, work, op, cpu_ctl, 1,
                                      next_host))
                                break;
                        vgpu_npu_process(npu);

                        uint64_t p = 0, o = 0, c = 0;
                        vgpu_npu_status_t s;
                        vgpu_npu_read_status(npu, &s);
                        for (int i = 0; i < 9; i++) {
                                if (i == NPU_RESULT_PIXELS) p = s.results[i];
                                if (i == NPU_RESULT_ALU_OPS) o = s.results[i];
                                if (i == NPU_RESULT_CULLED) c = s.results[i];
                        }
                        npu_pixels += p;
                        npu_ops += o;
                        npu_culled += c;

                        uint32_t *tmp = cur_host;
                        cur_host = next_host;
                        next_host = tmp;
                }

                double t1 = now_sec();
                double npu_time = t1 - t0;
                double npu_throughput = (double)npu_pixels / npu_time;

                printf("NPU  layers=%u work=%u pixels=%llu ops=%llu culled=%llu\n"
                       "       time=%.6f sec  throughput=%.3f Mpix/sec\n",
                       layers, work,
                       (unsigned long long)npu_pixels,
                       (unsigned long long)npu_ops,
                       (unsigned long long)npu_culled,
                       npu_time,
                       npu_throughput / 1e6);

                if (compare(cpu_out, npu_out, work))
                        printf("NPU vs CPU: MISMATCH\n");
                else
                        printf("NPU vs CPU: OUTPUT MATCH\n");
        }

        free(cpu_in);
        free(cpu_ctl);
        free(cpu_out);
        free(npu_out);
        vgpu_npu_destroy(npu);
        return 0;
}
