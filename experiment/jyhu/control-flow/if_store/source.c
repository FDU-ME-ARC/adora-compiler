#include <stdint.h>

void if_store(int condition, int32_t output[16]) {
  for (int i = 0; i < 16; ++i) {
    if (condition)
      output[i] = 7;
  }
}
