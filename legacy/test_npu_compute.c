/*
 * Minimal smoke test for the Voodoo4/Napalm + Wildcat Realizm-style NPU.
 *
 * This does a very small compute job two ways and prints both results:
 *   1) CPU reference implementation
 *   2) NPU via unit 1 compute + unit 0 texture upload as a data path
 *
 * It is intentionally tiny so it can be compiled and run as a stand-alone
 * program against the NPU module once the project is built.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "includes/private/video/vgpu_npu.h"

static void cpu_compute(uint32_t *out,
                      const uint32_t *in,
                      uint32_t count,
                      uint32_t op)
{
        for (uint32_t i = 0; i < count; i++) {
                uint32_t a = in[i];
                uint32_t b = in[i];
                uint32_t r = a;
                switch (op & 0xff) {
                case 0x01:
                        r = a + b;
                        break;
                case 0x02:
                        r = a * b;
                        break;
                case 0x03:
                        r = a ^ b;
                        break;
                case 0x04:
                        r = (a & b) | ((~a) & b);
                        break;
                case 0x05:
                        r = (a & b) | (a & ~b);
                        break;
                case 0x06:
                        r = (a & b) | ((~a) & b);
                        break;
                case 0x07:
                        r = (a & b) | (a & ~b);
                        break;
                default:
                        r = a;
                        break;
                }
                out[i] = r;
        }
}

static int check(const uint32_t *a, const uint32_t *b, uint32_t count)
{
        for (uint32_t i = 0; i < count; i++) {
                if (a[i] != b[i]) {
                        printf("mismatch at %u: cpu=%u npu=%u\n",
                               (unsigned)i, (unsigned)a[i], (unsigned)b[i]);
                        return 1;
                }
        }
        return 0;
}

int main(void)
{
        vgpu_npu_t *npu = vgpu_npu_create();
        if (!npu) {
                fprintf(stderr, "failed to create npu\n");
                return 1;
        }

        uint32_t vram_size = (uint32_t)vgpu_npu_vram_size(npu);
        uint32_t work = 1024;

        if (work * 4 * 2 > vram_size) {
                fprintf(stderr, "not enough npu ram for this test\n");
                vgpu_npu_destroy(npu);
                return 1;
        }

        uint32_t src_addr = 0;
        uint32_t dst_addr = work * 4;
        uint32_t ctl_addr = dst_addr + work * 4;
        uint32_t op = 0x01;

        /*
         * Build input on CPU first, then copy it into the NPU flat memory.
         */
        uint32_t *cpu_in = calloc(work, sizeof(uint32_t));
        uint32_t *cpu_out = calloc(work, sizeof(uint32_t));
        uint32_t *npu_out = calloc(work, sizeof(uint32_t));
        if (!cpu_in || !cpu_out || !npu_out) {
                free(cpu_in);
                free(cpu_out);
                free(npu_out);
                vgpu_npu_destroy(npu);
                return 1;
        }

        for (uint32_t i = 0; i < work; i++)
                cpu_in[i] = (i + 1) * 3;

        /* CPU reference. */
        cpu_compute(cpu_out, cpu_in, work, op);

        /*
         * Upload the same input into the NPU flat address space and run the
         * NPU compute path.
         */
        uint8_t *vram = vgpu_npu_vram_ptr(npu);
        memcpy(vram + src_addr, cpu_in, work * sizeof(uint32_t));

        vgpu_npu_unit1_compute_npu(npu, src_addr, dst_addr, work, op, ctl_addr, 0);
        vgpu_npu_process(npu);

        memcpy(npu_out, vram + dst_addr, work * sizeof(uint32_t));

        int failures = check(cpu_out, npu_out, work);
        if (!failures)
                printf("npu vs cpu: OK  work=%u op=0x%02x\n",
                       (unsigned)work, (unsigned)op);

        /*
         * Also exercise a NPU status readback so the engine can see results.
         */
        {
                vgpu_npu_status_t s;
                vgpu_npu_read_status(npu, &s);
                printf("npu status: busy=%d error=%d unit=%d\n",
                       s.busy, s.error, s.unit);
                for (int i = 0; i < 9; i++)
                        printf("  result[%d]=%llu\n", i, (unsigned long long)s.results[i]);
        }

        {
                uint64_t accum = vgpu_npu_unit1_read_result(npu, NPU_RESULT_ALU_OPS);
                printf("npu unit1 alu_ops=%llu\n", (unsigned long long)accum);
        }

        free(cpu_in);
        free(cpu_out);
        free(npu_out);
        vgpu_npu_destroy(npu);
        return failures;
}
