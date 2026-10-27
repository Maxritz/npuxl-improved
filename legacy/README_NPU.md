# Voodoo3/Wildcat NPU Virtualization - Experiments & Results

## Overview
This project implements a virtual NPU based on Voodoo3/Wildcat architecture with:
- Batched/tensor-style compute paths
- Voodoo, Wildcat, Merged, Shader GEMM, Tensor Core kernels
- Sherlock-style tracing instrumentation
- Bitmap-based memory management
- BeOS-style scheduler (teams/threads, priority-based preemptive scheduling, areas)
- 2-NPU cluster support

## Directory Structure
- `npu/` - Core NPU implementation
  - `virtual_npu.h/c` - Main NPU API and implementation
  - Various test programs
- `pcem/` - PCem emulator with Voodoo/NPU integration
- PDFs with hardware specs

## Test Results

### Single NPU vs CPU (512x512x512, optimized with -O3 -march=native)
| Kernel | Time (ms) | GFLOPS | Speedup vs CPU |
|--------|-----------|--------|----------------|
| CPU Baseline | ~63ms | ~4.24 GFLOPS | 1.00x |
| MERGED (Fused Tiled) | ~17.75ms | ~15.12 GFLOPS | 3.57x |
| TENSOR_CORE (16x16x16) | ~69.65ms | ~3.85 GFLOPS | 0.91x |
| SHADER_GEMM | slower | lower | < 1x |
| VOODOO | slower | lower | < 1x |

### 2-NPU Cluster vs CPU
| Configuration | Size | Time (ms) | GFLOPS | Speedup vs CPU |
|---------------|------|-----------|--------|----------------|
| CPU Baseline | 512³ | ~61.9ms | ~4.34 GFLOPS | 1.00x |
| 2-NPU Cluster (parallel split) | 512³ | ~14.3ms | ~18.77 GFLOPS | 4.33x |

### Key Optimizations
1. Fused tiled GEMM with aggressive blocking (128x128x128 tiles)
2. Loop unrolling (8-way unrolling in inner loops)
3. Compiler optimizations (-O3 -march=native -funroll-loops)
4. Better memory access patterns (B matrix accessed optimally)
5. Parallel cluster execution across 2 NPU instances

## Features Implemented
- **Instruction Set**: CPU (0), Voodoo (1), Wildcat (2), Merged (3), Shader GEMM (4), Shader Matmul (5), Fused GEMM (6), Tensor Core (7)
- **Memory Management**: Bitmap-based page tracking (4KB pages), BeOS-style areas with allocation/deallocation
- **Scheduler**: BeOS-style preemptive scheduler with teams/threads, priority-based selection (1-100), quantum allocation
- **Tracing**: Sherlock-IT style instrumentation for performance analysis

