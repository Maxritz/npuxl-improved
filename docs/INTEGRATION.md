# Engine integration guide

## 1. Choose an integration mode

| Mode | When to use | Pros | Cons |
|---|---|---|---|
| Link `-lvnpu` | you control the engine build | simplest, typed calls | rebuild needed for new lib |
| `dlopen` + `vnpu_get_backend_v1` | pluggable backend | zero build-time dependency | must resolve function table |
| `vnpu-node` over TCP | different process/machine or language | language-agnostic, weights resident | network latency |
| Legacy shim | old experiments | drop-in source compatibility | flat memory, synchronous |

## 2. Link directly

```c
#include "vnpu.h"

vnpu_config_t cfg;
vnpu_config_init(&cfg);
cfg.num_threads = 0;                 /* auto */
cfg.flags = VNPU_CONFIG_HIGH_THROUGHPUT;

vnpu_device_t *dev = NULL;
vnpu_stream_t *stream = NULL;
vnpu_device_create(&cfg, &dev);
vnpu_stream_create(dev, 0, &stream);

vnpu_matmul_f32_args_t mm = {0};
mm.A = A; mm.B = W; mm.bias = bias; mm.C = Y;
mm.M = tokens; mm.K = hidden; mm.N = out_dim;
mm.alpha = 1.0f; mm.beta = 0.0f;
mm.unit = VNPU_UNIT_AUTO;
vnpu_matmul_f32(stream, &mm);       /* submits + syncs */

vnpu_stream_destroy(stream);
vnpu_device_destroy(dev);
```

Compile:

```bash
cc -O2 -Iinclude engine.c -Llib -lvnpu -Wl,-rpath,'$ORIGIN/../lib' -o engine
```

On Windows, define `VNPU_USE_SHARED` before including `vnpu.h` and link
`vnpu.lib`.

## 3. Dynamic backend discovery

`vnpu_backend_v1_t` is a table of function pointers. It never owns memory and
never changes within a process.

```c
#include "vnpu_backend.h"
#include <dlfcn.h>

void *h = dlopen("libvnpu.so", RTLD_NOW);
const vnpu_backend_v1_t *(*get)(void) =
    (const vnpu_backend_v1_t *(*)(void))dlsym(h, "vnpu_get_backend_v1");
const vnpu_backend_v1_t *api = get();

if (api->abi_version != VNPU_ABI_VERSION) abort();
vnpu_device_t *dev;
api->device_create(NULL, &dev);
```

Check `api->table_size`, `api->backend_name` and individual pointers before
calling optional entries. New fields can be appended in reserved slots without
breaking v1 clients.

## 4. Mapping GGML / llama.cpp-style ops

A backend adapter receives engine tensors and emits virtual NPU commands. The
mapping is:

| Engine op | VNPU command |
|---|---|
| `GGML_OP_MUL_MAT` | `MATMUL_F32` or `MATMUL_I8` |
| `GGML_OP_RMS_NORM` | `RMSNORM_F32` |
| `GGML_OP_ROPE` | `ROPE_F32` |
| `GGML_OP_SOFT_MAX` | `SOFTMAX_F32` |
| `GGML_OP_SWIGLU` / SiLU + mul fusion | `SWIGLU_F32` |
| `GGML_OP_GELU` | `GELU_F32` |
| `GGML_OP_ADD` / residual | `ADD_F32` |
| `GGML_OP_FLASH_ATTN_EXT` | `ATTENTION_F32` |
| multi-token head | `MTP_HEAD_F32` |

Recommended adapter behavior:

1. Keep model weights in `vnpu_buffer_t` or packed INT8 weight objects.
2. Build a `std::vector<vnpu_command_t>` for one transformer layer.
3. Submit the whole batch to one stream with
   `vnpu_stream_submit(..., 1, event)`.
4. Sync once before sampling or KV-cache readback.
5. Use `trans_b=1` for decode weight matrices stored as `[N,K]`.
6. Fuse bias and epilogue into the GEMM rather than issuing separate commands.
7. Use `VNPU_UNIT_CELL` for `M == 1`, `WILDCAT` for masked/sparse work,
   `VOODOO` for prefill.

See `examples/engine_adapter.cpp` for a framework-neutral adapter skeleton.

## 5. INT8 weights

```c
vnpu_quantize_i8_args_t qw = {0};
qw.X = weight_f32; qw.Y = weight_i8; qw.scales = scale_w;
qw.rows = K; qw.cols = N; qw.axis = VNPU_QUANT_PER_COLUMN;
vnpu_quantize_i8(stream, &qw);

vnpu_packed_weights_t *packed = NULL;
vnpu_pack_weights_i8(dev, weight_i8, scale_w, K, N, 0, 0, &packed);

/* per token step */
vnpu_quantize_i8_args_t qa = {0};
qa.X = activation_f32; qa.Y = activation_i8; qa.scales = scale_a;
qa.rows = M; qa.cols = K; qa.axis = VNPU_QUANT_PER_ROW;
vnpu_quantize_i8(stream, &qa);

vnpu_matmul_i8_args_t mm = {0};
mm.A = activation_i8; mm.B = packed; mm.scale_a = scale_a;
mm.bias = bias; mm.C = output;
mm.M = M; mm.K = K; mm.N = N;
vnpu_matmul_i8(stream, &mm);
```

Weights should be quantized and packed once at model load. Activations are
quantized per row per step. For very large K, consider grouped quantization in a
future ABI revision.

## 6. Python

`bindings/python/vnpu.py` uses `ctypes` and the shared library. It exposes
`info()`, `matmul_f32()` and tracing helpers without requiring NumPy. With
NumPy installed, pass `.ctypes.data_as(POINTER(c_float))` pointers.

```python
from vnpu import VirtualNPU
npu = VirtualNPU()
npu.matmul(a_ptr, b_ptr, c_ptr, M, K, N)
print(npu.info())
```

## 7. Out-of-process mode

Start `vnpu-node` on the same box or another machine, then use
`vnpu_cluster_add_remote()` and `vnpu_cluster_cache_weights_f32()`. The engine
calls `vnpu_cluster_matmul_f32()` instead of `vnpu_matmul_f32()`. This is useful
for sandboxing, different language runtimes, or placing a draft model on
another host. See `docs/CLUSTER.md`.

## 8. Error handling and thread safety

- All status-returning functions return `vnpu_status_t`; use
  `vnpu_status_string()`.
- Never free a buffer, packed weight or event while a stream using it is
  pending.
- Distinct streams are thread safe. A single stream must be fed from one
  producer thread or externally serialized.
- Device destruction destroys remaining streams first; explicitly destroying
  streams is still preferred.
- `vnpu_command_t` pointers must remain valid until the associated event is
  signaled.

## 9. ABI stability

- `VNPU_ABI_VERSION` is 1.
- All public structs are POD and begin with versioned config/info.
- `vnpu_command_t` has 192 raw bytes in its union for growth.
- `vnpu_backend_v1_t` has eight reserved pointers.
- Deprecated fields are never repurposed; new ops get new opcodes.
