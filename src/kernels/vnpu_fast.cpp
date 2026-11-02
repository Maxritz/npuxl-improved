// vnpu_fast.cpp - direct AVX2+FMA GEMM fast path. No command layer, no stream hop,
// no tile vector, no trace string, no pool dispatch. Just the microkernel.
//
// API:
//   vnpu_gemm_f32_direct(A, B, C, M, K, N) - row-major, no transposes, alpha=1, beta=0
//   vnpu_gemm_i8_direct(A_q, B_packed, scale_a, C, M, K, N) - INT8 with packed weights
//
// These run on the calling thread with the 3x32 AVX2+FMA microkernel (F32) or the
// 2x32 INT8 microkernel. No heap allocation in the hot path.
#include "vnpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define VNPU_FAST_X86 1
#else
#define VNPU_FAST_X86 0
#endif

#if VNPU_FAST_X86 && (defined(__GNUC__) || defined(__clang__))
#define VNPU_FAST_AVX2 __attribute__((target("avx2,fma")))
#else
#define VNPU_FAST_AVX2
#endif

namespace {

// 3 rows x 32 cols, 12 accumulators + 3 broadcasts + 1 B vector = 16 YMM.
// Straight from kernels.cpp gemm_f32_avx2_nn_tile, minus the TileJob/params wrapper.
VNPU_FAST_AVX2
void gemm_f32_3x32(const float* A, int lda, const float* B, int ldb, float* C, int ldc,
                   int M, int K, int N) {
    constexpr int MR = 3, NR = 32;
    int m = 0;
    for (; m + MR <= M; m += MR) {
        const float* a0 = A + (size_t)m * lda;
        const float* a1 = A + (size_t)(m + 1) * lda;
        const float* a2 = A + (size_t)(m + 2) * lda;
        int n = 0;
        for (; n + NR <= N; n += NR) {
            __m256 c00 = _mm256_setzero_ps(), c01 = c00, c02 = c00, c03 = c00;
            __m256 c10 = _mm256_setzero_ps(), c11 = c10, c12 = c10, c13 = c10;
            __m256 c20 = _mm256_setzero_ps(), c21 = c20, c22 = c20, c23 = c20;
            for (int k = 0; k < K; ++k) {
                const float* brow = B + (size_t)k * ldb + n;
                __m256 b0 = _mm256_loadu_ps(brow + 0);
                __m256 va0 = _mm256_set1_ps(a0[k]);
                __m256 va1 = _mm256_set1_ps(a1[k]);
                __m256 va2 = _mm256_set1_ps(a2[k]);
                c00 = _mm256_fmadd_ps(va0, b0, c00);
                c10 = _mm256_fmadd_ps(va1, b0, c10);
                c20 = _mm256_fmadd_ps(va2, b0, c20);
                __m256 b1 = _mm256_loadu_ps(brow + 8);
                c01 = _mm256_fmadd_ps(va0, b1, c01);
                c11 = _mm256_fmadd_ps(va1, b1, c11);
                c21 = _mm256_fmadd_ps(va2, b1, c21);
                __m256 b2 = _mm256_loadu_ps(brow + 16);
                c02 = _mm256_fmadd_ps(va0, b2, c02);
                c12 = _mm256_fmadd_ps(va1, b2, c12);
                c22 = _mm256_fmadd_ps(va2, b2, c22);
                __m256 b3 = _mm256_loadu_ps(brow + 24);
                c03 = _mm256_fmadd_ps(va0, b3, c03);
                c13 = _mm256_fmadd_ps(va1, b3, c13);
                c23 = _mm256_fmadd_ps(va2, b3, c23);
            }
            float tmp[MR][NR];
            _mm256_storeu_ps(tmp[0] + 0, c00);
            _mm256_storeu_ps(tmp[0] + 8, c01);
            _mm256_storeu_ps(tmp[0] + 16, c02);
            _mm256_storeu_ps(tmp[0] + 24, c03);
            _mm256_storeu_ps(tmp[1] + 0, c10);
            _mm256_storeu_ps(tmp[1] + 8, c11);
            _mm256_storeu_ps(tmp[1] + 16, c12);
            _mm256_storeu_ps(tmp[1] + 24, c13);
            _mm256_storeu_ps(tmp[2] + 0, c20);
            _mm256_storeu_ps(tmp[2] + 8, c21);
            _mm256_storeu_ps(tmp[2] + 16, c22);
            _mm256_storeu_ps(tmp[2] + 24, c23);
            for (int i = 0; i < MR; ++i)
                for (int j = 0; j < NR; ++j)
                    C[(size_t)(m + i) * ldc + n + j] = tmp[i][j];
        }
        // tail columns: scalar
        for (; n < N; ++n) {
            float s0 = 0, s1 = 0, s2 = 0;
            for (int k = 0; k < K; ++k) {
                float b = B[(size_t)k * ldb + n];
                s0 += a0[k] * b;
                s1 += a1[k] * b;
                s2 += a2[k] * b;
            }
            C[(size_t)m * ldc + n] = s0;
            C[(size_t)(m + 1) * ldc + n] = s1;
            C[(size_t)(m + 2) * ldc + n] = s2;
        }
    }
    // tail rows: one row at a time, K-outermost for linear B walk
    for (; m < M; ++m) {
        const float* arow = A + (size_t)m * lda;
        for (int n = 0; n < N; ++n) C[(size_t)m * ldc + n] = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = arow[k];
            const float* brow = B + (size_t)k * ldb;
            float* crow = C + (size_t)m * ldc;
#if VNPU_FAST_X86
            int n = 0;
            __m256 va = _mm256_set1_ps(a);
            for (; n + 8 <= N; n += 8) {
                __m256 c = _mm256_loadu_ps(crow + n);
                __m256 b = _mm256_loadu_ps(brow + n);
                c = _mm256_fmadd_ps(va, b, c);
                _mm256_storeu_ps(crow + n, c);
            }
            for (; n < N; ++n) crow[n] += a * brow[n];
#else
            for (int n = 0; n < N; ++n) crow[n] += a * brow[n];
#endif
        }
    }
}

// Decode path: M small, stream B once. K-outermost, AVX2+FMA vectorized.
VNPU_FAST_AVX2
void gemm_f32_stream(const float* A, int lda, const float* B, int ldb, float* C, int ldc,
                     int M, int K, int N) {
    for (int m = 0; m < M; ++m) {
        const float* arow = A + (size_t)m * lda;
        float* crow = C + (size_t)m * ldc;
        for (int n = 0; n < N; ++n) crow[n] = 0.0f;
        for (int k = 0; k < K; ++k) {
            float a = arow[k];
            const float* brow = B + (size_t)k * ldb;
#if VNPU_FAST_X86
            int n = 0;
            __m256 va = _mm256_set1_ps(a);
            for (; n + 8 <= N; n += 8) {
                __m256 c = _mm256_loadu_ps(crow + n);
                __m256 b = _mm256_loadu_ps(brow + n);
                c = _mm256_fmadd_ps(va, b, c);
                _mm256_storeu_ps(crow + n, c);
            }
            for (; n < N; ++n) crow[n] += a * brow[n];
#else
            for (int n = 0; n < N; ++n) crow[n] += a * brow[n];
#endif
        }
    }
}

}  // namespace

extern "C" {

// Direct F32 GEMM: row-major A[M,K] x B[K,N] -> C[M,N]. No transpose, alpha=1, beta=0.
// Runs on the calling thread. No heap allocation. Picks 3x32 blocked (M>=3) or stream (M<3).
VNPU_API void VNPU_CALL vnpu_gemm_f32_direct(const float* A, const float* B, float* C,
                                            uint32_t M, uint32_t K, uint32_t N) {
    if (!A || !B || !C || !M || !K || !N) return;
    if (M >= 3)
        gemm_f32_3x32(A, (int)K, B, (int)N, C, (int)N, (int)M, (int)K, (int)N);
    else
        gemm_f32_stream(A, (int)K, B, (int)N, C, (int)N, (int)M, (int)K, (int)N);
}

}  // extern "C"
