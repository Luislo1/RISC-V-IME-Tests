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
#define notused a4
#define bp a5
#define borg a6
#define cp a7
#define mt t0
#define kt t3
#define nt t4
#define akp t5
#define bkp t6

sgemm_nn:
    mv borg, bp         # Store original b address
    li t0, 4            # Load auxiliar value to find iteration values
    div n, n, t0
    div m, m, t0
    li t0, 8
    div k, k, t0
	li kt, 0           	# Initialize inner loop counter
	li mt, 0           	# Initialize row loop counter (M)
	li nt, 0           	# Initialize column loop counter (N)

	# Set vector length for int8 (8 elements per vector)
	vsetvli x7, zero, e8, m1

	# Initialize result registers to zero
	vmv.v.i v0, 0
	vmv.v.i v1, 0
	vmv.v.i v8, 0
	vmv.v.i v9, 0

	# Outer loop over M (rows of A and C)
m_loop:
    vsetvli x7, zero, e8, m1
	# If we've processed all M rows, exit the loop
	bge mt, m, m_done  

	# Outer loop over N (columns of B and C)
n_loop:
	# If we've processed all N columns, exit the loop
    vsetvli x7, zero, e8, m1
	bge nt, n, n_done
    vmv.v.i v0, 0
	vmv.v.i v1, 0
    # vmv.v.i v2, 0
	# vmv.v.i v6, 0

    mv akp, ap          # Store original A pointer
	mv bkp, bp          # Store original B pointer

	# Reset K loop counter
	li kt, 0

	# Inner loop over K (shared dimension)
k_loop:
	# Check if K is finished
	bge kt, k, k_done  

	# Load A and B (assume A is row-major and B is row-major)
	vle8.v v2, (ap)   	# Load 4x8 elements from A
	vle8.v v6, (bp)   	# Load 8x4 elements from B

	# Perform matrix multiplication
	vmadot v8, v2, v6

	# Store result in v0, v1
	vsetvli x7, zero, e32, m1
	vadd.vv v0, v0, v8
	vadd.vv v1, v1, v9  

	# Restore v8 and v9 registers
	vmv.v.i v8, 0
	vmv.v.i v9, 0
	vsetvli x7, zero, e8, m1

	# Increment pointers to next 4x8 submatrix
	add ap, ap, 32    	# Move A pointer to next chunk 
	add bp, bp, 32    	# Move B pointer to next chunk

	addi kt, kt, 1    	# Increment K loop counter
	j k_loop

k_done:

    # Restore A and B pointers correctly for next row/column block
	mv ap, akp
	mv bp, bkp
	add bp, bp, 32      # Move B to next 8x4 block

	# Set vector length for storing int32 (result)
	vsetvli x7, zero, e32, m1
	mv t1, x7
	slli t1, t1, 2
	# Store results into C (matrix)
	vse32.v v0, (cp)  
	add cp, cp, t1   	# Update C pointer based on vector length set with vsetvli
	vse32.v v1, (cp)
    add cp, cp, t1 

	# Increment column counter for N loop
	addi nt, nt, 1
    
    

	j n_loop

n_done:

    # Move A to next row block and reset B
	add ap, ap, 32
	mv bp, borg     # Restore B address to first value

	# Increment row counter for M loop
	addi mt, mt, 1
	vmv.v.i v0, 0
	vmv.v.i v1, 0
    li nt, 0
	j m_loop

m_done:
	ret

