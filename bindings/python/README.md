# Python binding

`vnpu.py` is a zero-dependency `ctypes` wrapper around the shared library. It
does not require NumPy, though NumPy arrays work automatically.

```bash
make -j$(nproc)
export VNPU_LIBRARY=$PWD/lib/libvnpu.so
python3 bindings/python/vnpu.py
```

From your own code:

```python
import sys
sys.path.insert(0, "bindings/python")
from vnpu import VirtualNPU
from array import array

M, K, N = 64, 128, 64
A = array("f", [0.01]) * (M * K)
B = array("f", [0.02]) * (K * N)
C = array("f", [0.0]) * (M * N)

with VirtualNPU(threads=0, trace=False) as npu:
    print(npu.info())
    npu.matmul(A, B, C, M, K, N, unit=1)
    print(C[0])
```

## Notes

- `matmul()` uses the synchronous convenience API, which submits one command and
  waits. For many operations, use the C command-stream API from a C++ extension.
- `unit=0` is AUTO, `1` Voodoo, `2` Wildcat, `3` Cell.
- `trans_b=True` expects B as `[N,K]`.
- Pass `bias`, `alpha`, `beta`, `epilogue` as needed.
- `trace_chrome("trace.json")` writes a Chrome Trace JSON file.
- Set `VNPU_FORCE_ISA=scalar` when debugging with AddressSanitizer.
