#include <riscv_vector.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define N 32

#define MAX_BLOCKSIZE 32
#define MLEN 4
#define KLEN 8
#define NLEN 4
#define OUTPUT_LEN 16

float a_array[MAX_BLOCKSIZE] = {
	-99.0,  52.0,  114.0,  -63.0,  -41.0,  127.0,   65.0,  -18.0,
	102.0,  77.0,  -8.0,  -123.0,   36.0,  29.0,  -95.0,   49.0,
	-110.0,  -74.0,  56.0,   11.0,   -2.0,  63.0,  -12.0,  -81.0,
	104.0,  58.0,  -26.0,  9.0,  110.0,  -47.0,  32.0,  -99.0
};

float b_array[MAX_BLOCKSIZE] = {
	-13.0,   51.0,   92.0,  -115.0,   88.0,  -76.0,   11.0,   50.0,
	33.0,   79.0,  -122.0,   72.0,  102.0,   -89.0,  -59.0,  38.0,
	-25.0,  96.0,   19.0,   -6.0,   -89.0,  -108.0,  14.0,   59.0,
	56.0,  120.0,   -90.0,   -71.0,  -5.0,   44.0,  97.0,  103.0
};

float golden_array[OUTPUT_LEN];
float c_array[OUTPUT_LEN];

void sgemm_golden() {
  for (size_t i = 0; i < MLEN; ++i)
    for (size_t j = 0; j < NLEN; ++j)
      for (size_t k = 0; k < KLEN; ++k)
        golden_array[i * NLEN + j] += a_array[i * KLEN + k] * b_array[j + k * NLEN];
}


// reference https://github.com/riscv/riscv-v-spec/blob/master/example/sgemm.S
// c += a*b (alpha=1, no transpose on input matrices)
// matrices stored in C row-major order
extern void sgemm_nn(size_t size_m, size_t size_n, size_t size_k,
               const float *a, // m * k matrix
               size_t lda,
               const float *b, // k * n matrix
               size_t ldb,
               float *c, // m * n matrix
               size_t ldc); 

int main() {
  // golden
  memcpy(golden_array, b_array, OUTPUT_LEN * sizeof(float));
  sgemm_golden();
  // vector
  memcpy(c_array, b_array, OUTPUT_LEN * sizeof(float));
  sgemm_nn(MLEN, NLEN, KLEN, a_array, KLEN, b_array, NLEN, c_array, NLEN);

  int pass = 1;
  for (int i = 0; i < OUTPUT_LEN; i++) {
    if (golden_array[i] != c_array[i]) {
      printf("index %d fail, %d=!%d\n", i, golden_array[i], c_array[i]);
      pass = 0;
    }
    else {
      printf("index %d pass, %d==%d\n", i, golden_array[i], c_array[i]); 
    }
  }
  if (pass)
    printf("pass\n");
  return (pass == 0);
}
