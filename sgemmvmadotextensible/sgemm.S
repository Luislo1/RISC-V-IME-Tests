    .text
	.balign 4
	.global sgemm_nn
# RV64IDV system

# Function Signature:
# void sgemm_nn(size_t n, size_t m, size_t k, const int8_t* a, size_t lda, const int8_t* b, size_t ldb, int8_t* c, size_t ldc)
# c += a * b (alpha=1, no transpose)

#define n a0
#define m a1
#define k a2
#define ap a3
#define iterations a4
#define bp a5
#define bstride a6
#define cp a7
#define cstride t0
#define nt t2
#define kt t3
#define cnp t4
#define akp t5
#define bkp s0
#define nvl s1
#define ccp s2

sgemm_nn:
	li kt, 0 # Initialize inner loop counter

	# Set vector length for int8
	vsetvli x7, zero, e8, m1

	# Initialize result registers to zero
	vmv.v.i v0, 0
	vmv.v.i v1, 0
    vmv.v.i v8, 0
	vmv.v.i v9, 0

    # For K multiples of 4
k_loop:
	# Check if K is finished
	bge kt, iterations, k_done  

	# Load A and B (assume A row-major and B row-major)
	vle8.v v2, (ap)  	# Load 4x8 elements from A
	vle8.v v6, (bp)  	# Load 4x8 elements from B

	# Perform matrix multiplication
	vmadot v8, v2, v6

    # Store result in v0, v1
	vsetvli x7, zero, e32, m1
    vadd.vv v0, v0, v8
    vadd.vv v1, v1, v9  

    # Restore v8 and v9 vectors
    vmv.v.i v8, 0
	vmv.v.i v9, 0
	vsetvli x7, zero, e8, m1

	# Increment pointers to next 4x8 submatrix
	add ap, ap, 32    	# Move A pointer to next chunk
	add bp, bp, 32    	# Move B pointer to next chunk

	addi kt, kt, 1   	# Increment loop counter
	j k_loop


k_done:
	# Set vector length for storing int32
	vsetvli x7, zero, e32, m1 
    mv t1, x7
    slli t1, t1, 2
	# Store results with proper address update
	vse32.v v0, (cp)  	 
	add cp, cp, t1   # Update cp based on actual vector length
	vse32.v v1, (cp)  

	ret


