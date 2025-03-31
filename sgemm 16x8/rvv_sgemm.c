#include <riscv_vector.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define N 32

#define MAX_BLOCKSIZE 128
#define MLEN 16
#define KLEN 8
#define NLEN 16
#define OUTPUT_LEN 256

float a_array[MAX_BLOCKSIZE] = {
	-0.433, -1.666,  0.125,  0.288, -1.146,  1.191,  1.189, -0.038,  
 	0.327,  0.175, -0.187,  0.726, -0.588,  2.183, -0.136,  0.114,  
 	1.067,  0.059, -0.096, -0.832,  0.294, -1.336,  0.714,  1.624,  
	-0.692,  0.858,  1.254, -1.594, -1.441,  0.571, -0.400,  0.100,  
 	0.673,  1.235, -0.924,  0.584, -1.123,  0.789, -0.457,  1.988,  
	-0.346,  1.457, -1.235,  0.346,  1.877, -0.568,  0.679, -1.099,  
 	0.321, -0.765,  1.432,  0.987, -1.654,  0.876, -0.543,  1.321,  
	-0.987,  0.654, -1.210,  0.432,  1.789, -0.876,  0.543, -0.321,  
 	0.210, -1.432,  0.876, -0.765,  1.543, -0.987,  0.432, -1.321,  
 	0.543, -1.210,  0.876, -0.654,  1.432, -0.543,  0.765, -1.321,  
	-0.987,  1.210, -0.876,  0.543, -1.432,  0.987, -0.321,  1.210,  
	-0.654,  1.543, -0.876,  0.432, -1.321,  0.543, -1.210,  0.987,  
 	0.654, -1.432,  0.876, -0.765,  1.543, -0.987,  0.432, -1.321,  
 	0.543, -1.210,  0.876, -0.654,  1.432, -0.543,  0.765, -1.321,  
	-0.987,  1.210, -0.876,  0.543, -1.432,  0.987, -0.321,  1.210,
	-0.987,  1.210, -0.876,  0.543, -1.432,  0.987, -0.321,  1.210   
};


float b_array[MAX_BLOCKSIZE] = {
 	1.749,  0.133,  0.325, -0.794,  0.315, -0.527,  0.932,  1.165,  
	-2.046, -0.644,  1.741,  0.487,  1.049,  1.489,  1.271, -1.856,  
 	2.134,  1.436, -0.917, -1.106,  0.811,  0.699, -0.402,  1.269,  
	-0.784,  0.213,  0.788,  0.897, -0.187,  1.013,  0.248,  0.100,  
	-1.234,  0.987, -0.543,  1.432, -0.765,  0.876, -1.321,  0.543,  
 	1.210, -0.876,  0.654, -1.432,  0.987, -0.321,  1.543, -0.876,  
 	0.432, -1.321,  0.765, -0.543,  1.210, -0.987,  0.876, -0.654,  
 	1.432, -0.765,  0.543, -1.210,  0.987, -0.876,  0.321, -1.543,  
 	0.876, -1.432,  0.543, -0.765,  1.321, -0.432,  0.987, -1.210,  
	-0.876,  1.543, -0.765,  0.543, -1.432,  0.654, -0.987,  1.321,  
 	0.876, -0.543,  1.210, -0.765,  0.432, -1.543,  0.987, -0.876,  
 	1.321, -0.654,  1.432, -0.543,  0.765, -1.210,  0.987, -0.876,  
	-0.543,  1.432, -0.765,  0.876, -1.321,  0.543, -1.210,  0.987,  
 	0.654, -1.543,  0.432, -0.876,  1.321, -0.765,  0.543, -1.210,  
 	0.987, -0.876,  1.432, -0.543,  0.765, -1.321,  0.876, -0.654,
0.987, -0.876,  1.432, -0.543,  0.765, -1.321,  0.876, -0.654  
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
               size_t ldc); //{
//  size_t vl;
//  for (size_t m = 0; m < size_m; ++m) {
//    const float *b_n_ptr = b;
//    float *c_n_ptr = c;
//    for (size_t c_n_count = size_n; c_n_count; c_n_count -= vl) {
//      vl = __riscv_vsetvl_e32m1(c_n_count );
//      const float *a_k_ptr = a;
//      const float *b_k_ptr = b_n_ptr;
//      vfloat32m1_t acc = __riscv_vle32_v_f32m1(c_n_ptr, vl);
//      for (size_t k = 0; k < size_k; ++k) {
//        vfloat32m1_t b_n_data = __riscv_vle32_v_f32m1(b_k_ptr, vl);
//        acc = __riscv_vfmacc_vf_f32m1(acc, *a_k_ptr, b_n_data, vl);
//        b_k_ptr += ldb;
//        a_k_ptr++;
//      }
//      __riscv_vse32_v_f32m1(c_n_ptr, acc, vl);
//      c_n_ptr += vl;
//      b_n_ptr += vl;
//    }
//    a += lda;
//    c += ldc;
//  }
//}

int fp_eq(float reference, float actual, float relErr)
{
  // if near zero, do absolute error instead.
  float absErr = relErr * ((fabsf(reference) > relErr) ? fabsf(reference) : relErr);
  return fabsf(actual - reference) < absErr;
}

int main() {
  // golden
  memset(golden_array, 0, OUTPUT_LEN * sizeof(float));
  sgemm_golden();
  // vector
  memset(c_array, 0, OUTPUT_LEN * sizeof(float));
  sgemm_nn(MLEN, NLEN, KLEN, a_array, KLEN, b_array, NLEN, c_array, NLEN);

  int pass = 1;
  for (int i = 0; i < OUTPUT_LEN; i++) {
    if (!fp_eq(golden_array[i], c_array[i], 1e-5)) {
      printf("index %d fail, %f=!%f\n", i, golden_array[i], c_array[i]);
      pass = 0;
    }
  }
  if (pass)
    printf("pass\n");
  return (pass == 0);
}
