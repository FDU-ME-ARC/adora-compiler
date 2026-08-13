#include <stdint.h>

void nested_if(int a, int b, int32_t output[16]) {
  for (int i = 0; i < 16; ++i) {
    if (a) {
      if (b)
        output[i] = 1;
    }
  }
}
