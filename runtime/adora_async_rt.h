// adora_async_rt.h — ADORA async-event runtime ABI (PR3).
//
// C ABI; intentionally kept minimal so backend ports (CGRA, CUDA, Vulkan)
// can drop in a different implementation without touching the IR.
//
// streamId is reserved for PR4 multi-stream support; pass 0 for single-stream.
#ifndef ADORA_ASYNC_RT_H
#define ADORA_ASYNC_RT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AdoraEvent_ *AdoraEvent;

// Allocate a new event. Returns NULL on failure.
AdoraEvent adoraEventCreate(void);

// Release an event previously created by adoraEventCreate.
void adoraEventDestroy(AdoraEvent e);

// Signal `e` on stream `streamId`: any thread waiting on `e` will unblock
// once this call completes.
void adoraEventRecord(AdoraEvent e, int streamId);

// Block the calling thread (stream `streamId`) until `e` is signalled.
void adoraEventWait(AdoraEvent e, int streamId);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ADORA_ASYNC_RT_H
