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
#define astride a4
#define bp a5
#define bstride a6
#define cp a7
#define cstride t0
#define nt t2
#define bnp t3
#define cnp t4
#define akp t5
#define bkp s0
#define nvl s1
#define ccp s2


sgemm_nn:
	# Set vector length for int8 (for loading A & B)
	vsetvli x7, zero, e8, m1

	# Load A and B with stride
	vle8.v v14, (ap) # Load matrix A
	# vlse8.v v16, (bp), bstride   # Load matrix B
	vle8.v v16, (bp)   # Load matrix B

	# Initialize result registers to zero
	vmv.v.i v0, 0
	vmv.v.i v1, 0

	# Compute matrix multiplication
	vmadot v0, v14, v16  

	# Set vector length for storing int32
	vsetvli x7, zero, e32, m1 
    mv t1, x7
    slli t1, t1, 2
	# Store results with proper address update
	vse32.v v0, (cp)  	 
	add cp, cp, t1   # Update cp based on actual vector length
	vse32.v v1, (cp)  	 

	ret

