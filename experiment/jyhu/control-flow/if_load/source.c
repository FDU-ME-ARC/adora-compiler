#include <stdint.h>

void if_load(int condition, int32_t input[16], int32_t output[16]) {
  for (int i = 0; i < 16; ++i)
    output[i] = condition ? input[i] : 0;
}
