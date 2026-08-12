#include <stdint.h>

void if_else(int condition, int32_t output[16]) {
  for (int i = 0; i < 16; ++i)
    output[i] = condition ? 11 : 22;
}
