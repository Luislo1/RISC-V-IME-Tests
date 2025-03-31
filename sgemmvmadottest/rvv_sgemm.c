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
	-99,   52,  114,  -63,  -41,  127,   65,  -18, 102,   77,   -8, -123,   36,   29,  -95,   49,
    -110,  -74,   56,   11,   -2,   63,  -12,  -81, 104,   58,  -26,	9,  110,  -47,   32,  -99
};

int8_t b_array[MAX_BLOCKSIZE] = {
	-13,   51,   92, -115,   88,  -76,   11,   50,   45,  -82,  103,   27,   98,   34,  -55,   61,
 	33,   79, -122,   72,  102,  -89,  -59,   38,   12,   67,   -7,  -48,  120,   29,   -3,   95
};

int8_t b_array[MAX_BLOCKSIZE] = {
	-13,   51,   92, -115,   88,  -76,   11,   50,   45,  -82,  103,   27,   98,   34,  -55,   61,
 	33,   79, -122,   72,  102,  -89,  -59,   38,   12,   67,   -7,  -48,  120,   29,   -3,   95,
    //siguiente bloque
	-25,   96,   19,   -6,  -89, -108,   14,   59,   83,  -17,   65,  -92,   53,  111,   28,  -41,
 	56,  120,  -90,  -71,   -5,   44,   97,  103,   -2,   75,  -60,   19,   87,  -21,   72,  -99
};

int8_t b_array_transposed4[MAX_BLOCKSIZE] = {
	-13,   88,   45,  98,   33,  102,   12,   120,
	51,   -76,  -82,   34,  79,   -89,  67,  29,
	92,  11,   103,   -55,   -122,  -59,  -7,   -3,
	-115,  50,   27,   61,  72,   38,  -48,  95
};

int8_t b_array_transposedtest[MAX_BLOCKSIZE] = { //Hay que hacer la transpuesta en bloques de 4x8.
	-13,   88,   45,  98,   33,  102,   12,   120,
	51,   -76,  -82,   34,  79,   -89,  67,  29,
	92,  11,   103,   -55,   -122,  -59,  -7,   -3,
	-115,  50,   27,   61,  72,   38,  -48,  95,
    //siguiente transpuesta
    -25, -89, 83, 53, 56, -5, -2, 87, 
    96, -108, -17, 111, 120, 44, 75, -21,
    19, 14, 65, 28, -90, 97, -60, 72,
    -6, 59, -92, -41, -71, 103, 19, -99
};

int8_t b_array_transposed16[MAX_BLOCKSIZE] = {
	-13, 33, -25, 56,
    51, 79, 96, 120,
    92, -122, 19, -90,
    -115, 72, -6, -71,
    88, 102, -89, -5,
    -76, -89, -108, 44,
    11, -59, 14, 97,
    50, 38, 59, 103,
    45, 12, 83, -2,
    -82, 67, -17, 75,
    103, -7, 65, -60,
    27, -48, -92, 19,
    98, 120, 53, 87,
    34, 29, 111, -21,
    -55, -3, 28, 72,
    61, 95, -41, -99
};

int8_t b_array_transposed8[MAX_BLOCKSIZE] = {
	-13, 45, 33, 12, -25, 83, 56, -2,
    51, -82, 79, 67, 96, -17, 56, 75,
    92, 103, -122, -7, 19, 65, -90, -60,
    -115, 27, 72, -48, -6, -92, -71, 19,
    88, 98, 102, 120, -89, 53, -5, 87,
    -76, 34, -89, 29, -108, 111, 44, -21,
    11, -55, -59, -3, 14, 28, 97, 72,
    50, 61, 38, 95, 59, -41, 103, -99
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
