#include <riscv_vector.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define N 32

#define MAX_BLOCKSIZE 64
#define MLEN 4
#define KLEN 16
#define NLEN 4
#define OUTPUT_LEN 16


int8_t a_array[MAX_BLOCKSIZE] = { //row-major
	-99,   52,  114,  -63,  -41,  127,   65,  -18,   24,  -88,   37,   92,   15,  -56,  111,   -7,
	102,   77,   -8, -123,   36,   29,  -95,   49,  -54,   80,  -99,  117,  -43,  109,   60,  -20,
    //siguiente bloque
   -110,  -74,   56,   11,   -2,   63,  -12,  -81,   90,   42,  -68,   74,   25,  -35,   99,  -55,
	104,   58,  -26,	9,  110,  -47,   32,  -99,   67,  -22,   84,   -9,  -77,  126,   -3,   45
};

int8_t a_array_transposed[MAX_BLOCKSIZE] = { //Como vmadot utiliza bloques de 4x8, es necesario reordenar la matriz en memoria para que esten los dos bloques en orden. Cogemos los elementos [0-7], [16-23], [32-39] y [48-55] y los ponemos como primera matriz de 32 elementos contiguos.
	-99,   52,  114,  -63,  -41,  127,   65,  -18, 102,   77,   -8, -123,   36,   29,  -95,   49,
    -110,  -74,   56,   11,   -2,   63,  -12,  -81, 104,   58,  -26,	9,  110,  -47,   32,  -99,
    //siguiente transpuesta
    24,  -88,   37,   92,   15,  -56,  111,   -7, -54,   80,  -99,  117,  -43,  109,   60,  -20,
    90,   42,  -68,   74,   25,  -35,   99,  -55, 67,  -22,   84,   -9,  -77,  126,   -3,   45
};



int8_t b_array[MAX_BLOCKSIZE] = { //column-major
	-13, 88, 45, 98, 
    33, 102, 12, 120, 
    -25, -89, 83, 53, 
    56, -5, -2, 87,
    51, -76, -82, 34, 
    79, -89, 67, 29, 
    96, -108, -17, 111, 
    120, 44, 75, -21,
    //siguiente bloque
    92, 11, 103, -55, 
    -122, -59, -7, -3, 
    19, 14, 65, 28, 
    -90, 97, -60, 72,
    -115, 50, 27, 61, 
    72, 38, -48, 95, 
    -6, 59, -92, -41, 
    -71, 103, 19, -99
};

int8_t b_array_transposed[MAX_BLOCKSIZE] = { //Hay que hacer la transpuesta en bloques de 4x8.
	-13, 33, -25, 56, 51, 79, 96, 120,
    88, 102, -89, -5, -76, -89, -108, 44,
    45, 12, 83, -2, -82, 67, -17, 75,
    98, 120, 53, 87, 34, 29, 111, -21,
    //siguiente transpuesta
    92, -122, 19, -90, -115, 72, -6, -71,
    11, -59, 14, 97, 50, 38, 59, 103,
    103, -7, 65, -60, 27, -48, -92, 19,
    -55, -3, 28, 72, 61, 95, -41, -99

};

int32_t golden_array[OUTPUT_LEN];
int32_t c_array[OUTPUT_LEN];

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

void reorder_column_to_row(const int8_t* B_col_major, int8_t* B_row_major, size_t rows, size_t cols) {
	for (size_t col = 0; col < cols; ++col) {
    	for (size_t row = 0; row < rows; ++row) {
        	// Source index (column-major): B[row + col * rows]
        	// Destination index (row-major): B[row * cols + col]
        	B_row_major[row * cols + col] = B_col_major[col * rows + row];
    	}
	}
}



int main() {
  // golden
  memset(golden_array, 0, OUTPUT_LEN * sizeof(int32_t));
  sgemm_golden();
  // vector
  memset(c_array, 0, OUTPUT_LEN * sizeof(int32_t));
  int iterations = KLEN / 8;
  sgemm_nn(MLEN, NLEN, KLEN, a_array_transposed, iterations, b_array_transposed, NLEN, c_array, NLEN);
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
