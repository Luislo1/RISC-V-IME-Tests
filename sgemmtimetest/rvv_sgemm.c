#include <riscv_vector.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

int MLEN, KLEN, NLEN;
int OUTPUT_LEN;

int8_t *a_array;
int8_t *a_array_packed;
int8_t *b_array;
int8_t *b_array_packed_and_transposed;
int32_t *golden_array;
int32_t *c_array;
int32_t *c_array_tr;

void sgemm_golden() {
	for (size_t i = 0; i < MLEN; ++i)
    	for (size_t j = 0; j < NLEN; ++j)
        	for (size_t k = 0; k < KLEN; ++k) {
            	golden_array[i * NLEN + j] += a_array[i * KLEN + k] * b_array[j + k * NLEN];
        	}
}

extern void sgemm_nn(size_t size_m, size_t size_n, size_t size_k,
                  	const int8_t *a,
                  	size_t lda,
                  	const int8_t *b,
                  	size_t ldb,
                  	int32_t *c,
                  	size_t ldc);

void reorder_matrix_copy(const int32_t *src, int32_t *dst, size_t blocks_amount_rows, size_t blocks_amount_column) {
	for (int l = 0; l < blocks_amount_column; l++) {
    	for (int i = 0; i < blocks_amount_rows; i++) {
        	for (int j = 0; j < 4; j++) {
            	for (int k = 0; k < 4; k++) {
                	dst[j * (4 * blocks_amount_rows) + k + i * 4 + l * (4 * 4 * blocks_amount_rows)] =
                       	src[j * 4 + k + i * 16 + l * (4 * 4 * blocks_amount_rows)];
            	}
        	}
    	}
	}
}

void generate_random_matrix(int8_t *matrix, int rows, int cols, int seed) {
	srand(seed);
	for (int i = 0; i < rows * cols; i++) {
    	matrix[i] = (int8_t)(rand() % 201 - 100); // Values between -100 and 100
	}
}

void pack_matrix(int8_t *src, int8_t *dst, size_t blocks_amount_rows, size_t blocks_amount_column) {
	for (int l = 0; l < blocks_amount_column; l++) {
    	for (int i = 0; i < blocks_amount_rows; i++) {
        	for (int j = 0; j < 4; j++) {
            	for (int k = 0; k < 8; k++) {
                	dst[j * 8 + k + i * 8 * 4 + l * (8 * 4 * blocks_amount_rows)] =
                       	src[j * (8 * blocks_amount_rows) + k + i * 8 + l * (8 * 4 * blocks_amount_rows)];
            	}
        	}
    	}
	}
}

void pack_and_transpose_matrix(int8_t *src, int8_t *dst, size_t blocks_amount_rows, size_t blocks_amount_column) {

	for (int i = 0; i < blocks_amount_rows; i++) {
        for (int l = 0; l < blocks_amount_column; l++) {
        	for (int j = 0; j < 4; j++) {
            	for (int k = 0; k < 8; k++) {
                	dst[j * 8 + k + i * 8 * 4 + l * (8 * 4 * blocks_amount_rows)] =
                       	src[j + k * (blocks_amount_rows * 4) + i * 4 + l * 8 * 4 * blocks_amount_rows]; //cambiar l e i en funcion de si quieres transponer en bloques por columna o por fila (nos interesa fila)
            	}
        	}
    	}
	}
}

void print_matrix(int8_t *matrix, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ",matrix[i * cols + j]);
        }
        printf("\n");
	}
}

int main(int argc, char *argv[]) {
//	printf("Enter MLEN, KLEN, NLEN, and seed: ");
//	int seed;
//	scanf("%d %d %d %d", &MLEN, &KLEN, &NLEN, &seed);
    if (argc != 5) {
        perror("Usage: ./sgemmIMERandom MLEN>=4 KLEN>=8 NLEN>=4 SEED");
        return EXIT_FAILURE;    
    }
    MLEN = atoi(argv[1]);
    KLEN = atoi(argv[2]);
    NLEN = atoi(argv[3]);
    int seed = atoi(argv[4]);


	OUTPUT_LEN = MLEN * NLEN;
    clock_t start = clock();  
	a_array = (int8_t *)malloc(MLEN * KLEN * sizeof(int8_t));
	b_array = (int8_t *)malloc(KLEN * NLEN * sizeof(int8_t));
	//golden_array = (int32_t *)calloc(OUTPUT_LEN, sizeof(int32_t));
	c_array = (int32_t *)calloc(OUTPUT_LEN, sizeof(int32_t));
	//c_array_tr = (int32_t *)calloc(OUTPUT_LEN, sizeof(int32_t));
    
	generate_random_matrix(a_array, MLEN, KLEN, seed);
	generate_random_matrix(b_array, KLEN, NLEN, seed + 1);
    clock_t end = clock();
	printf("Time spent initializing everything (no packing): %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
    //printf("Matrix A\n");
    //print_matrix(a_array, MLEN, KLEN);
    //printf("Matrix B\n");
    //print_matrix(b_array, KLEN, NLEN);
    
    

    //a_array_packed = (int8_t *)malloc(MLEN * KLEN * sizeof(int8_t));
    //pack_matrix(a_array, a_array_packed, KLEN / 8, MLEN / 4);

	//b_array_packed_and_transposed = (int8_t *)malloc(KLEN * NLEN * sizeof(int8_t));
    //pack_and_transpose_matrix(b_array, b_array_packed_and_transposed, NLEN / 4,  KLEN / 8);

    //printf("Matrix A packed\n");
    //print_matrix(a_array_packed, MLEN, KLEN);
    //printf("Matrix B packed and transposed\n");
    //print_matrix(b_array_packed_and_transposed, KLEN, NLEN);
    //clock_t start = clock();
	//sgemm_golden();
    //clock_t end = clock();
	//printf("sgemm_golden execution time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    start = clock();
	sgemm_nn(NLEN, MLEN, KLEN, a_array, 7, b_array, NLEN, c_array, NLEN);
    end = clock();
	printf("sgemm_nn execution time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

	//reorder_matrix_copy(c_array, c_array_tr, NLEN / 4, MLEN / 4);

	//int pass = 1;
	//for (int i = 0; i < OUTPUT_LEN; i++) {
    //	if (golden_array[i] != c_array_tr[i]) {
    //    	printf("index %d fail, %d != %d\n", i, golden_array[i], c_array_tr[i]);
    //    	pass = 0;
    //	}// else {
        //	printf("index %d pass, %d == %d\n", i, golden_array[i], c_array_tr[i]);
    	//}
	//}
	//if (pass)
    //	printf("pass\n");

	free(a_array);
	free(b_array);
	free(b_array_packed_and_transposed);
	free(golden_array);
	free(c_array);
	free(c_array_tr);
    
	return 0;
}

