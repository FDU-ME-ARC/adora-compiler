// adora_async_rt.c — Reference implementation of the ADORA async-event ABI.
//
// Uses POSIX mutex + condition variable for portable host-side synchronisation.
// Real CGRA / GPU backends replace this file with hardware-specific calls while
// keeping the same ABI declared in adora_async_rt.h.
//
// Thread-safety: each AdoraEvent is independently thread-safe; concurrent calls
// to adoraEventRecord and adoraEventWait on the same event are safe.

#include "adora_async_rt.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>

struct AdoraEvent_ {
  atomic_int     signaled; // 0 = not yet signaled, 1 = signaled
  pthread_cond_t  cv;
  pthread_mutex_t mu;
};

AdoraEvent adoraEventCreate(void) {
  AdoraEvent e = (AdoraEvent)calloc(1, sizeof(*e));
  if (!e) return NULL;
  atomic_init(&e->signaled, 0);
  pthread_cond_init(&e->cv, NULL);
  pthread_mutex_init(&e->mu, NULL);
  return e;
}

void adoraEventDestroy(AdoraEvent e) {
  if (!e) return;
  pthread_cond_destroy(&e->cv);
  pthread_mutex_destroy(&e->mu);
  free(e);
}

// Signal: set signaled=1, broadcast to wake all waiting threads.
void adoraEventRecord(AdoraEvent e, int streamId) {
  (void)streamId; // reserved for PR4 multi-stream
  if (!e) return;
  pthread_mutex_lock(&e->mu);
  atomic_store(&e->signaled, 1);
  pthread_cond_broadcast(&e->cv);
  pthread_mutex_unlock(&e->mu);
}

// Wait: block until signaled==1.
void adoraEventWait(AdoraEvent e, int streamId) {
  (void)streamId; // reserved for PR4 multi-stream
  if (!e) return;
  pthread_mutex_lock(&e->mu);
  while (!atomic_load(&e->signaled))
    pthread_cond_wait(&e->cv, &e->mu);
  pthread_mutex_unlock(&e->mu);
}
