# Examples

## `dynamic_load.c`

Pure C example that loads `libvnpu.so` with `dlopen()`, resolves
`vnpu_get_backend_v1()`, creates a device/stream and runs a small GEMM without
linking against the library at build time.

```bash
make -j$(nproc)
cc -O2 -Iinclude examples/dynamic_load.c -ldl -o /tmp/dynamic_load
VNPU_LIBRARY=./lib/libvnpu.so /tmp/dynamic_load
```

On Windows, replace `dlopen`/`dlsym` with `LoadLibrary`/`GetProcAddress` and
define `VNPU_USE_SHARED`.

## `engine_adapter.cpp`

A framework-neutral adapter skeleton showing how a GGML/llama.cpp-style engine
would map tensors and ops to a batch of `vnpu_command_t` packets. It is
intentionally free of engine headers so it can be copied into a backend plugin.

Key ideas:

- one stream per inference session;
- one command batch per transformer layer;
- packed INT8 weights reused across tokens;
- fused bias/epilogue inside `MATMUL_*`;
- one sync before sampling, not after every op.

See `docs/INTEGRATION.md` for the full op mapping table.
