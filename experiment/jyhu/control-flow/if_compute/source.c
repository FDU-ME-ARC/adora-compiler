#include <stdint.h>

void if_compute(int condition, int32_t input[16], int32_t output[16]) {
  for (int i = 0; i < 16; ++i) {
    int32_t value;
    if (condition)
      value = input[i] * 2 + 1;
    else
      value = input[i] + 3;
    output[i] = value;
  }
}
