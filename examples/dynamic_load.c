/*
 * dynamic_load.c - use Virtual NPU without linking against it.
 *
 * cc -O2 -Iinclude examples/dynamic_load.c -ldl -o /tmp/dynamic_load
 * VNPU_LIBRARY=./lib/libvnpu.so /tmp/dynamic_load
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vnpu.h"
#include "vnpu_backend.h"

typedef const vnpu_backend_v1_t *(*get_backend_fn)(void);
typedef void (*config_init_fn)(vnpu_config_t *);

static float *alloc_f32(size_t n, float value)
{
    float *p = NULL;
    if (posix_memalign((void **)&p, 64, n * sizeof(float)) != 0)
        return NULL;
    for (size_t i = 0; i < n; ++i) p[i] = value;
    return p;
}

int main(void)
{
    const char *path = getenv("VNPU_LIBRARY");
    if (!path) path = "libvnpu.so";
    void *lib = dlopen(path, RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "dlopen(%s): %s\n", path, dlerror());
        return 1;
    }
    get_backend_fn get_backend = (get_backend_fn)dlsym(lib,
        "vnpu_get_backend_v1");
    config_init_fn config_init = (config_init_fn)dlsym(lib,
        "vnpu_config_init");
    if (!get_backend || !config_init) {
        fprintf(stderr, "missing backend symbols\n");
        dlclose(lib);
        return 1;
    }
    const vnpu_backend_v1_t *api = get_backend();
    if (!api || api->abi_version != VNPU_ABI_VERSION || !api->device_create ||
        !api->matmul_f32 || !api->status_string) {
        fprintf(stderr, "incompatible backend vtable\n");
        dlclose(lib);
        return 1;
    }

    vnpu_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    config_init(&cfg);
    cfg.num_threads = 0; /* auto */
    cfg.flags = VNPU_CONFIG_HIGH_THROUGHPUT;

    vnpu_device_t *dev = NULL;
    vnpu_stream_t *stream = NULL;
    vnpu_status_t rc = api->device_create(&cfg, &dev);
    if (rc != VNPU_OK) goto fail;
    rc = api->stream_create(dev, 0, &stream);
    if (rc != VNPU_OK) goto fail;

    vnpu_device_info_t info;
    api->device_get_info(dev, &info);
    printf("loaded %s from %s, %u threads\n", api->backend_name,
           info.backend, info.num_threads);

    const uint32_t M = 8, K = 16, N = 8;
    float *A = alloc_f32((size_t)M * K, 1.0f);
    float *B = alloc_f32((size_t)K * N, 2.0f);
    float *C = alloc_f32((size_t)M * N, 0.0f);
    if (!A || !B || !C) {
        rc = VNPU_ERR_OUT_OF_MEMORY;
        goto cleanup;
    }
    vnpu_matmul_f32_args_t mm;
    memset(&mm, 0, sizeof(mm));
    mm.A = A; mm.B = B; mm.C = C;
    mm.M = M; mm.K = K; mm.N = N;
    mm.alpha = 1.0f; mm.beta = 0.0f;
    mm.unit = VNPU_UNIT_AUTO;
    rc = api->matmul_f32(stream, &mm);
    if (rc == VNPU_OK) {
        /* Every C element should be K * 1 * 2 = 32. */
        printf("C[0]=%.1f C[last]=%.1f expected %.1f\n",
               C[0], C[(size_t)M * N - 1], 2.0f * K);
        if (C[0] != 2.0f * K || C[(size_t)M * N - 1] != 2.0f * K)
            rc = VNPU_ERR_INTERNAL;
    }
cleanup:
    free(A); free(B); free(C);
    if (stream) api->stream_destroy(stream);
fail:
    if (dev) api->device_destroy(dev);
    if (rc != VNPU_OK)
        fprintf(stderr, "failed: %s\n", api->status_string(rc));
    dlclose(lib);
    return rc == VNPU_OK ? 0 : 1;
}
