/* npu/virtual_npu.h — batched/tensor compute NPU (CPU / Voodoo / Wildcat). */

#ifndef VIRTUAL_NPU_H
#define VIRTUAL_NPU_H

#include <stdint.h>
#include <stddef.h>

typedef struct virtual_npu virtual_npu_t;

virtual_npu_t *virtual_npu_create(size_t vram_bytes);
void            virtual_npu_destroy(virtual_npu_t *n);

uint8_t *virtual_npu_memory(virtual_npu_t *n);
size_t   virtual_npu_memory_size(virtual_npu_t *n);void virtual_npu_set_memory_bitmap(virtual_npu_t *n, int enable);
uint64_t virtual_npu_get_bitmap_mask(virtual_npu_t *n, size_t page);
/* BeOS-style scheduler and memory management */
int virtual_npu_create_team(virtual_npu_t *n, const char *name);
int virtual_npu_create_thread(virtual_npu_t *n, int team, int priority);
void virtual_npu_reschedule(virtual_npu_t *n);
void *virtual_npu_create_area(virtual_npu_t *n, const char *name, size_t size, int lock);
int virtual_npu_delete_area(virtual_npu_t *n, void *addr);



typedef struct virtual_npu_status {
        int busy;
        int error;
        uint64_t ops;
        uint64_t processed;
        uint64_t culled;
        uint64_t dispatches;
        uint64_t batches;
} virtual_npu_status_t;

void virtual_npu_read_status(virtual_npu_t *n, virtual_npu_status_t *s);
void virtual_npu_reset_acc(virtual_npu_t *n);
void virtual_npu_invalidate(virtual_npu_t *n);
void virtual_npu_reset(virtual_npu_t *n);
void virtual_npu_dump(virtual_npu_t *n);

/*
 * Batched tensor compute job: C = A * B
 *
 * Memory layout in NPU flat space (row-major, 4-byte elements):
 *   A at a_addr,  shape [M, K],   size M*K  elements
 *   B at b_addr,  shape [K, N],   size K*N  elements
 *   C at c_addr,  shape [M, N],   size M*N  elements, output
 *   ctl at ctl_addr, shape [M, N], size M*N elements, optional cull control
 *
 * op_kind selects the compute style:
 *   0 = CPU direct tiled matmul
 *   1 = Voodoo batched tile dispatch (command-stream style)
 *   2 = Wildcat batched tile dispatch (programmable compute unit style,
 *                                    with optional hierarchical-Z-style tile culling)
 *
 * Tiles are walked with triangle clamping at the matrix edges, so partial
 * tiles at the right/bottom/back borders are handled correctly.
 *
 * Cull control uses a per-output-element value when cull_threshold != 0.
 *   - CPU path: per-element cull
 *   - Voodoo path: per-element cull
 *   - Wildcat path: per-tile cull (a tile is skipped if its control value is
 *     below threshold)
 */
enum {
        NPU_OP_CPU = 0,
        NPU_OP_VOODOO = 1,
        NPU_OP_WILDCAT = 2,
        NPU_OP_MERGED = 3,
        NPU_OP_SHADER_GEMM = 4,
        NPU_OP_SHADER_MATMUL = 5,
        NPU_OP_FUSED_GEMM = 6,
        NPU_OP_TENSOR_CORE = 7,
};

void virtual_npu_compute(virtual_npu_t *n,
                        size_t a_addr,
                        size_t b_addr,
                        size_t c_addr,
                        size_t ctl_addr,
                        size_t M,
                        size_t K,
                        size_t N,
                        size_t tile_m,
                        size_t tile_k,
                        size_t tile_n,
                        uint32_t op_kind,
                        uint32_t op_code,
                        uint32_t cull_threshold);

void virtual_npu_process(virtual_npu_t *n);

#endif

/* Sherlock tracing */
enum {
    SHERLOCK_TRAP_DISPATCH = 0,
    SHERLOCK_TRAP_MEMORY = 1,
    SHERLOCK_TRAP_INSTRUCTION = 2,
    SHERLOCK_TRAP_SYNC = 3,
    SHERLOCK_TRAP_ALLOC = 4,
    SHERLOCK_TRAP_TRANSFER = 5,
    SHERLOCK_TRAP_DEPENDENCY = 6,
    SHERLOCK_TRAP_ACCURACY = 7,
    SHERLOCK_TRAP_COMPUTE = 8,
};

typedef struct sherlock_trace {
    double dispatch_time;
    double memory_time;
    double compute_time;
    double sync_time;
    double total_time;
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t cache_misses;
    uint64_t ops_count;
    int traps_enabled;
} sherlock_trace_t;

void sherlock_enable_tracing(int enable);
void sherlock_reset_trace(void);
void sherlock_dump_trace(const char *label);
void sherlock_get_trace(sherlock_trace_t *trace);
void sherlock_trace_start(virtual_npu_t *n, int trap_type, const char *name);
void sherlock_trace_end(virtual_npu_t *n, int trap_type, uint64_t ops, uint64_t bytes);

