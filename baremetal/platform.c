#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/crt/logging.h"
#include "tvm/runtime/crt/page_allocator.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

// ------------------------------------------------------------------
// Memory Management (Updated for TVM CRT)
// ------------------------------------------------------------------

// Returns an error code, puts the pointer in *out_ptr
tvm_crt_error_t TVMPlatformMemoryAllocate(size_t num_bytes, DLDevice dev,
                                          void **out_ptr) {
    if (num_bytes == 0) {
        *out_ptr = NULL;
        return kTvmErrorNoError;
    }

    void *ptr = malloc(num_bytes);
    if (ptr == NULL) {
        return kTvmErrorPlatformNoMemory;
    }

    *out_ptr = ptr;
    return kTvmErrorNoError;
}

tvm_crt_error_t TVMPlatformMemoryFree(void *ptr, DLDevice dev) {
    free(ptr);
    return kTvmErrorNoError;
}

// ------------------------------------------------------------------
// Error Handling
// ------------------------------------------------------------------
void TVMPlatformAbort(tvm_crt_error_t error_code) {
    fprintf(stderr, "TVM Platform Abort: Error code %d\n", error_code);
    exit(1);
}

// ------------------------------------------------------------------
// Logging
// ------------------------------------------------------------------
void TVMLogf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stdout, fmt, args);
    va_end(args);
}

// ------------------------------------------------------------------
// Timers (Required stubs)
// ------------------------------------------------------------------
tvm_crt_error_t TVMPlatformTimerStart() { return kTvmErrorNoError; }
tvm_crt_error_t TVMPlatformTimerStop(double *elapsed_time_seconds) {
    return kTvmErrorNoError;
}
