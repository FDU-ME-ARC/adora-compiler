#include <stdint.h>

void if_simple(int condition, int32_t output[16]) {
  for (int i = 0; i < 16; ++i) {
    if (condition)
      output[i] = output[i] + 1;
  }
}
