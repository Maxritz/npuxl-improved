"""
Minimal ctypes binding for libvnpu.

No NumPy is required.  Buffers may be:
  * ctypes float arrays / pointers,
  * objects exposing ``.ctypes.data_as`` (NumPy),
  * Python ``array('f')`` objects,
  * raw integer addresses.

Example:
    from array import array
    from vnpu import VirtualNPU
    npu = VirtualNPU()
    A = array('f', [1.0]) * (M*K)
    B = array('f', [2.0]) * (K*N)
    C = array('f', [0.0]) * (M*N)
    npu.matmul(A, B, C, M, K, N)
    print(npu.info())
"""
from __future__ import annotations

import ctypes
import json
import os
from array import array
from ctypes import (
    POINTER, Structure, byref, c_char, c_char_p, c_float, c_int, c_uint32,
    c_uint64, c_void_p, cast
)

__all__ = ["VirtualNPU", "Config", "DeviceInfo", "MatmulF32Args", "find_library"]


def find_library(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("VNPU_LIBRARY")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "..", "lib", "libvnpu.so"),
        os.path.join(here, "..", "..", "lib", "libvnpu.dylib"),
        os.path.join(here, "..", "..", "bin", "vnpu.dll"),
        "libvnpu.so",
        "libvnpu.dylib",
        "vnpu.dll",
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return "libvnpu.so"


class Config(Structure):
    _fields_ = [
        ("abi_version", c_uint32),
        ("num_threads", c_uint32),
        ("num_compute_units", c_uint32),
        ("tile_m", c_uint32),
        ("tile_k", c_uint32),
        ("tile_n", c_uint32),
        ("flags", c_uint32),
        ("reserved0", c_uint32),
        ("trace_file", c_char_p),
        ("node_name", c_char_p),
    ]


class DeviceInfo(Structure):
    _fields_ = [
        ("abi_version", c_uint32),
        ("name", c_char * 64),
        ("vendor", c_char * 64),
        ("cpu_arch", c_char * 32),
        ("backend", c_char * 64),
        ("simd_bits", c_uint32),
        ("num_threads", c_uint32),
        ("num_compute_units", c_uint32),
        ("local_store_bytes", c_uint32),
        ("tile_m", c_uint32),
        ("tile_k", c_uint32),
        ("tile_n", c_uint32),
        ("reserved0", c_uint32),
        ("features", c_uint64),
        ("l1d_bytes", c_uint64),
        ("l2_bytes", c_uint64),
        ("l3_bytes", c_uint64),
        ("total_memory_bytes", c_uint64),
    ]


class MatmulF32Args(Structure):
    _fields_ = [
        ("A", c_void_p),
        ("B", c_void_p),
        ("bias", c_void_p),
        ("C", c_void_p),
        ("M", c_uint32),
        ("K", c_uint32),
        ("N", c_uint32),
        ("lda", c_uint32),
        ("ldb", c_uint32),
        ("ldc", c_uint32),
        ("trans_a", c_uint32),
        ("trans_b", c_uint32),
        ("alpha", c_float),
        ("beta", c_float),
        ("epilogue", c_uint32),
        ("unit", c_uint32),
        ("reserved0", c_uint32),
        ("tile_mask", c_void_p),
        ("mask_tile_m", c_uint32),
        ("mask_tile_n", c_uint32),
    ]


def _float_ptr(obj):
    if obj is None:
        return None
    if isinstance(obj, int):
        return cast(c_void_p(obj), POINTER(c_float))
    if hasattr(obj, "ctypes") and hasattr(obj.ctypes, "data_as"):
        return obj.ctypes.data_as(POINTER(c_float))
    if isinstance(obj, array):
        buf = (c_float * len(obj)).from_buffer(obj)
        return cast(buf, POINTER(c_float))
    if isinstance(obj, ctypes.Array):
        return cast(obj, POINTER(c_float))
    nbytes = getattr(obj, "nbytes", len(obj) * ctypes.sizeof(c_float))
    count = nbytes // ctypes.sizeof(c_float)
    buf = (c_float * count).from_buffer(obj)
    return cast(buf, POINTER(c_float))


class VirtualNPU:
    """Thin Python wrapper around a device and one synchronous stream."""

    def __init__(self, library: str | None = None, threads: int = 0,
                 trace: bool = False, trace_file: str | None = None,
                 high_throughput: bool = True):
        path = find_library(library)
        self.lib = ctypes.CDLL(path)
        self.path = path
        self._bind()
        cfg = Config()
        self.lib.vnpu_config_init(byref(cfg))
        cfg.num_threads = threads
        if trace:
            cfg.flags |= 1  # VNPU_CONFIG_TRACE
        if high_throughput:
            cfg.flags |= 1 << 6  # VNPU_CONFIG_HIGH_THROUGHPUT
        self._trace_file = trace_file.encode() if trace_file else None
        if self._trace_file:
            cfg.trace_file = self._trace_file
        self.device = c_void_p()
        rc = self.lib.vnpu_device_create(byref(cfg), byref(self.device))
        self._check(rc)
        self.stream = c_void_p()
        rc = self.lib.vnpu_stream_create(self.device, 0, byref(self.stream))
        self._check(rc)

    def _bind(self):
        L = self.lib
        L.vnpu_config_init.argtypes = [POINTER(Config)]
        L.vnpu_config_init.restype = None
        L.vnpu_device_create.argtypes = [POINTER(Config), POINTER(c_void_p)]
        L.vnpu_device_create.restype = c_int
        L.vnpu_device_destroy.argtypes = [c_void_p]
        L.vnpu_device_destroy.restype = None
        L.vnpu_device_get_info.argtypes = [c_void_p, POINTER(DeviceInfo)]
        L.vnpu_device_get_info.restype = c_int
        L.vnpu_stream_create.argtypes = [c_void_p, c_int, POINTER(c_void_p)]
        L.vnpu_stream_create.restype = c_int
        L.vnpu_stream_destroy.argtypes = [c_void_p]
        L.vnpu_stream_destroy.restype = None
        L.vnpu_stream_sync.argtypes = [c_void_p]
        L.vnpu_stream_sync.restype = c_int
        L.vnpu_matmul_f32.argtypes = [c_void_p, POINTER(MatmulF32Args)]
        L.vnpu_matmul_f32.restype = c_int
        L.vnpu_trace_enable.argtypes = [c_void_p, c_int]
        L.vnpu_trace_enable.restype = c_int
        L.vnpu_trace_reset.argtypes = [c_void_p]
        L.vnpu_trace_reset.restype = None
        L.vnpu_trace_dump.argtypes = [c_void_p, c_char_p]
        L.vnpu_trace_dump.restype = c_int
        L.vnpu_trace_write_chrome.argtypes = [c_void_p, c_char_p]
        L.vnpu_trace_write_chrome.restype = c_int
        L.vnpu_status_string.argtypes = [c_int]
        L.vnpu_status_string.restype = c_char_p
        L.vnpu_version_string.argtypes = []
        L.vnpu_version_string.restype = c_char_p

    def _check(self, rc: int):
        if rc != 0:
            msg = self.lib.vnpu_status_string(rc)
            raise RuntimeError(msg.decode() if msg else f"vnpu error {rc}")

    def version(self) -> str:
        value = self.lib.vnpu_version_string()
        return value.decode() if value else "unknown"

    def info(self) -> dict:
        info = DeviceInfo()
        self._check(self.lib.vnpu_device_get_info(self.device, byref(info)))
        def s(x):
            return x.split(b"\0", 1)[0].decode(errors="replace")
        return {
            "version": self.version(),
            "library": self.path,
            "name": s(info.name),
            "vendor": s(info.vendor),
            "arch": s(info.cpu_arch),
            "backend": s(info.backend),
            "simd_bits": info.simd_bits,
            "threads": info.num_threads,
            "compute_units": info.num_compute_units,
            "tiles": [info.tile_m, info.tile_k, info.tile_n],
            "features": info.features,
            "l1d_bytes": info.l1d_bytes,
            "l2_bytes": info.l2_bytes,
            "l3_bytes": info.l3_bytes,
            "memory_bytes": info.total_memory_bytes,
        }

    def matmul(self, A, B, C, M: int, K: int, N: int,
               bias=None, alpha: float = 1.0, beta: float = 0.0,
               trans_b: bool = False, unit: int = 0, epilogue: int = 0):
        """C[M,N] = alpha * A[M,K] * B[K,N] + bias + beta*C, then epilogue."""
        args = MatmulF32Args()
        args.A = cast(_float_ptr(A), c_void_p)
        args.B = cast(_float_ptr(B), c_void_p)
        args.bias = cast(_float_ptr(bias), c_void_p) if bias is not None else None
        args.C = cast(_float_ptr(C), c_void_p)
        args.M, args.K, args.N = M, K, N
        args.alpha, args.beta = alpha, beta
        args.trans_b = 1 if trans_b else 0
        args.unit = unit
        args.epilogue = epilogue
        self._check(self.lib.vnpu_matmul_f32(self.stream, byref(args)))

    def trace_enable(self, enabled: bool = True):
        self._check(self.lib.vnpu_trace_enable(self.device, 1 if enabled else 0))

    def trace_reset(self):
        self.lib.vnpu_trace_reset(self.device)

    def trace_dump(self, path: str | None = None):
        self._check(self.lib.vnpu_trace_dump(
            self.device, path.encode() if path else None))

    def trace_chrome(self, path: str):
        self._check(self.lib.vnpu_trace_write_chrome(self.device, path.encode()))

    def sync(self):
        self._check(self.lib.vnpu_stream_sync(self.stream))

    def close(self):
        if getattr(self, "stream", None) and self.stream.value:
            self.lib.vnpu_stream_destroy(self.stream)
            self.stream = c_void_p()
        if getattr(self, "device", None) and self.device.value:
            self.lib.vnpu_device_destroy(self.device)
            self.device = c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    M, K, N = 32, 48, 24
    A = array("f", [float((i % 17) - 8) * 0.01 for i in range(M * K)])
    B = array("f", [float((i % 13) - 6) * 0.02 for i in range(K * N)])
    C = array("f", [0.0]) * (M * N)
    with VirtualNPU(trace=True) as npu:
        print(json.dumps(npu.info(), indent=2))
        npu.matmul(A, B, C, M, K, N)
        print("C[0] =", C[0])
        npu.trace_dump()
