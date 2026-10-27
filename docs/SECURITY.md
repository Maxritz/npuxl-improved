# Security considerations

Virtual NPU v1 is designed for trusted CPU-only environments. It is a compute
runtime, not a hardened network service.

## Local library

- The C ABI trusts pointers and shapes supplied by the embedding engine.
  Validate model files before passing tensors to the runtime.
- Commands are POD; an engine must keep referenced buffers alive until the
  associated event is signaled.
- Shape overflow checks exist, but extremely large allocations can still cause
  `std::bad_alloc`, which is returned as `VNPU_ERR_OUT_OF_MEMORY`.
- `VNPU_FORCE_ISA` is a debugging feature; do not rely on it for security.

## Network node

`vnpu-node` uses a compact binary TCP protocol with a shared token.

Current protections:

- token must be sent in the first `OP_HELLO` message;
- payload size is capped at 1 GiB;
- K/M/N shape multiplication is overflow-checked;
- unknown opcodes return `VNPU_ERR_UNSUPPORTED`;
- connections have send/receive timeouts.

Current limitations:

- **No TLS.** Traffic, including the token, is plaintext.
- Token comparison is not constant-time hardened.
- No rate limiting, quotas, or per-client memory limits.
- Native-endian wire format: use matching architectures or add explicit
  serialization.
- No replay protection or message authentication codes.

## Deployment recommendations

1. Bind to a private interface or localhost.
2. Put nodes behind WireGuard, Tailscale, an SSH tunnel, or an mTLS proxy.
3. Use a long random token and rotate it.
4. Firewall the node port; never expose it to the public internet.
5. Run the node as an unprivileged user with memory/CPU limits.
6. Monitor `OP_GET_STATS` and Sherlock traces for anomalous request sizes.
7. For multi-tenant use, add authentication, quotas and TLS termination before
   the node.

## Reporting issues

If you find a memory-safety or protocol issue, include:

- CPU features from `vnpu-cli info --json`;
- the exact command or API sequence;
- a Sherlock trace if reproducible;
- whether `VNPU_FORCE_ISA=scalar` avoids the issue.
