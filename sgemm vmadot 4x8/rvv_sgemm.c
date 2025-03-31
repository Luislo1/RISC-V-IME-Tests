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

int8_t a_array[MAX_BLOCKSIZE] = {
	-99,  52,  114,  -63,  -41,  127,   65,  -18,
	102,  77,  -8,  -123,   36,  29,  -95,   49,
	-110,  -74,  56,   11,   -2,  63,  -12,  -81,
	104,  58,  -26,  9,  110,  -47,  32,  -99
};

int8_t b_array[MAX_BLOCKSIZE] = {
	-13,   51,   92,  -115,   88,  -76,   11,   50,
	33,   79,  -122,   72,  102,   -89,  -59,  38,
	-25,  96,   19,   -6,   -89,  -108,  14,   59,
	56,  120,   -90,   -71,  -5,   44,  97,  103
};

int8_t b_array_transposed4[MAX_BLOCKSIZE] = {
	-13,   88,   33,  102,   -25,  -89,   56,   -5,
	51,   -76,  79,   -89,  96,   -108,  120,  44,
	92,  11,   -122,   -59,   19,  14,  -90,   97,
	-115,  50,   72,   38,  -6,   59,  -71,  103
};

int8_t b_array_transposed8[MAX_BLOCKSIZE] = {
	-13,   33,   -25,  56,
    51,  79,   96,   120,
	92,   -122,  19,   -90,  
    -115,   72,  -6,  -71,
	88,  102,   -89,   -5,   
    -76,  -89,  -108,   44,
	11,  -59,   14,   97,  
    50,   38,  59,  103
};

int32_t golden_array[OUTPUT_LEN];
int32_t c_array[OUTPUT_LEN];

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
               const int8_t *a, // m * k matrix
               size_t lda,
               const int8_t *b, // k * n matrix
               size_t ldb,
               int32_t *c, // m * n matrix
               size_t ldc);

int main() {
  // golden
  memset(golden_array, 0, OUTPUT_LEN * sizeof(int32_t));
  sgemm_golden();
  // vector
  memset(c_array, 0, OUTPUT_LEN * sizeof(int32_t));
  sgemm_nn(MLEN, NLEN, KLEN, a_array, KLEN, b_array_transposed4, NLEN, c_array, NLEN);

  int pass = 1;
  for (int i = 0; i < OUTPUT_LEN; i++) {
    if (golden_array[i] != c_array[i]) {
      printf("index %d fail, %d=!%d\n", i, golden_array[i], c_array[i]);
      pass = 0;
    }
  }
  if (pass)
    printf("pass\n");
  return (pass == 0);
}
