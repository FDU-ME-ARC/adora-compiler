#include <stdint.h>

void if_elseif_else(int a, int b, int32_t output[16]) {
  for (int i = 0; i < 16; ++i) {
    if (a)
      output[i] = 11;
    else if (b)
      output[i] = 22;
    else
      output[i] = 33;
  }
}
