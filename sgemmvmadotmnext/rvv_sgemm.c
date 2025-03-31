#include <riscv_vector.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#define N 32

#define MAX_BLOCKSIZE 64
#define MLEN 8
#define KLEN 8
#define NLEN 8
#define OUTPUT_LEN 64


int8_t a_array[MAX_BLOCKSIZE] = { //Al ser una matriz 8x8, como vmadot usa chunks de 4x8, no es necesario realizar ninguna operacion en la matriz al pasarlo al codigo ensamblador.
	-99,   52,  114,  -63,  -41,  127,   65,  -18,
    24,  -88,   37,   92,   15,  -56,  111,   -7,
	102,   77,   -8, -123,   36,   29,  -95,   49,
    -54,   80,  -99,  117,  -43,  109,   60,  -20,
    //siguiente bloque
    -110,  -74,   56,   11,   -2,   63,  -12,  -81,
    90,   42,  -68,   74,   25,  -35,   99,  -55,
	104,   58,  -26,	9,  110,  -47,   32,  -99,
    67,  -22,   84,   -9,  -77,  126,   -3,   45
};


int8_t b_array[MAX_BLOCKSIZE] = { //column-major
	-13, 88, 45, 98, 33, 102, 12, 120, 
    -25, -89, 83, 53, 56, -5, -2, 87,
    51, -76, -82, 34, 79, -89, 67, 29, 
    96, -108, -17, 111, 120, 44, 75, -21,
    //siguiente bloque
    92, 11, 103, -55, -122, -59, -7, -3, 
    19, 14, 65, 28, -90, 97, -60, 72,
    -115, 50, 27, 61, 72, 38, -48, 95, 
    -6, 59, -92, -41, -71, 103, 19, -99
};

int8_t b_array_transposed[MAX_BLOCKSIZE] = { //Transpuesta de acuerdo a la foto en el documento
	-13, -25, 51, 96, 92, 19, -115, -6,
    88, -89, -76, -108, 11, 14, 50, 59,
    45, 83, -82, -17, 103, 65, 27, -92,
    98, 53, 34, 111, -55, 28, 61, -41,
    //siguiente transpuesta
    33, 56, 79, 120, -122, -90, 72, -71,
    102, -5, -89, 44, -59, 97, 38, 103,
    12, -2, 67, 75, -7, -60, -48, 19,
    120, 87, 29, -21, -3, 72, 95, -99
};

int32_t golden_array[OUTPUT_LEN];
int32_t c_array[OUTPUT_LEN];
int32_t c_array_tr[OUTPUT_LEN];

void sgemm_golden() {
  for (size_t i = 0; i < MLEN; ++i)
    for (size_t j = 0; j < NLEN; ++j)
      for (size_t k = 0; k < KLEN; ++k) {
        golden_array[i * NLEN + j] += a_array[i * KLEN + k] * b_array[j + k * NLEN];
      }
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

void reorder_matrix_copy(const int32_t *src, int32_t *dst, size_t blocks_amount_rows, size_t blocks_amount_column)
{
    int k = 0;
    for (int l = 0; l < blocks_amount_column; l++) { //For each column
        for (int i = 0; i < blocks_amount_rows; i++) { //For each block in each row
	        for (int j = 0; j < 4; j++) { //4x4 block process
                for (int k = 0; k < 4; k++) {
                    memcpy(&dst[j * (4 * blocks_amount_rows) + k + i *4 + l * (4 * 4 * blocks_amount_rows)], 
                        &src[j * 4 + k + i * 16 + l * (4 * 4 * blocks_amount_rows)], sizeof(int32_t));
                }
            }
        }
    }
}

int main() {
  // golden
  memset(golden_array, 0, OUTPUT_LEN * sizeof(int32_t));
  sgemm_golden();
  // vector
  memset(c_array, 0, OUTPUT_LEN * sizeof(int32_t));
  memset(c_array_tr, 0, OUTPUT_LEN * sizeof(int32_t));
  int iterations = KLEN / 8;
  sgemm_nn(MLEN, NLEN, KLEN, a_array, iterations, b_array_transposed, NLEN, c_array, NLEN);
  reorder_matrix_copy(c_array, c_array_tr, NLEN / 4, MLEN / 4);
  int pass = 1;
  for (int i = 0; i < OUTPUT_LEN; i++) {
    if (golden_array[i] != c_array_tr[i]) {
      printf("index %d fail, %d=!%d\n", i, golden_array[i], c_array_tr[i]);
      pass = 0;
    }
    if (golden_array[i] == c_array_tr[i]) {
      printf("index %d pass, %d=!%d\n", i, golden_array[i], c_array_tr[i]);
    }
  }
  if (pass)
    printf("pass\n");
  return (pass == 0);
}
