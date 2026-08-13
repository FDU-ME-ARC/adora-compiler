#include <stdint.h>

void loop_if(int32_t input[16], int32_t output[16]) {
  int32_t sum = 0;
  for (int i = 0; i < 16; ++i) {
    if (input[i] > 0)
      sum += input[i];
    output[i] = sum;
  }
}
