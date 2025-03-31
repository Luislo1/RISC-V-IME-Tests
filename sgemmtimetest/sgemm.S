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
#define seven a4
#define bp a5
#define borg a6
#define cp a7
#define mt t0
#define kt t3
#define nt t4
#define akp t5
#define bkp t6

sgemm_nn:
    mv borg, bp         # Store initial b address
    li t0, 4            # Load auxiliar value to find iteration values. N / 4 & M / 4 (due to 4x8x4 MAC)
    div n, n, t0
    div m, m, t0
    li t0, 8            # Load auxiliar value to find iteration values. K / 8 (due to 4x8x4 MAC)
    div k, k, t0
	li kt, 0           	# Initialize inner loop counter
	li mt, 0           	# Initialize row loop counter (M)
	li nt, 0           	# Initialize column loop counter (N)

	# Outer loop over M (rows of A and C)
m_loop:
	# If we've processed all M rows, exit the loop
	bge mt, m, m_done  

	# Outer loop over N (columns of B and C)
n_loop:
	# If we've processed all N columns, exit the loop. Set VL for int8 elements first
    vsetvli x7, zero, e8, m1
	bge nt, n, n_done
    
    # Reset accumulation registers to zero
    vmv.v.i v0, 0
	vmv.v.i v1, 0

    mv akp, ap          # Store original A pointer
	mv bkp, bp          # Store original B pointer

	# Reset K loop counter
	li kt, 0
    # Prepare bp shift amount to go to next 4x4 row in B (shift N columns)
    li t1, 32 
    mul t1, t1, n       

k_loop_choose:
    # Check if K is finished
	bge kt, k, k_done
    li seven, 7
    sub t2, k, kt  
    bge t2, seven, k_loop7
    addi seven, seven, -3   
    bge t2, seven, k_loop4  
    addi seven, seven, -2    
    bge t2, seven, k_loop2  


k_loop:

	# Load A and B (assume A is row-major and B is row-major)
	vle8.v v2, (ap)   	# Load 4x8 elements from A
	vle8.v v3, (bp)   	# Load 8x4 elements from B

	# Perform matrix multiplication
	vmadot v16, v2, v3

	# Store result in v0, v1
	vsetvli x7, zero, e32, m1
	vadd.vv v0, v0, v16
	vadd.vv v1, v1, v17  

	# Restore vmadot registers
	vmv.v.i v16, 0
	vmv.v.i v17, 0
	vsetvli x7, zero, e8, m1

	# Increment pointers to next 4x8 submatrices
	add ap, ap, 32    	# Move A pointer to next chunk
	add bp, bp, t1    	# Move B pointer to next chunk (We must go to the next column here)

	addi kt, kt, 1    	# Increment K loop counter
	j k_loop_choose


k_loop2:

	# Load A and B (assume A is row-major and B is row-major)
	vle8.v v2, (ap)   	# Load 4x8 elements from A
	vle8.v v3, (bp)   	# Load 8x4 elements from B
	add ap, ap, 32    	# Move A pointer to next chunk
	add bp, bp, t1    	# Move B pointer to next chunk (We must go to the next column here)
    vle8.v v4, (ap)   	# Load 4x8 elements from A
	vle8.v v5, (bp)   	# Load 8x4 elements from B

	# Perform matrix multiplication
	vmadot v16, v2, v3
	vmadot v18, v4, v5

	# Store result in v0, v1
	vsetvli x7, zero, e32, m1
	vadd.vv v0, v0, v16
	vadd.vv v1, v1, v17  
	vadd.vv v0, v0, v18
	vadd.vv v1, v1, v19

	# Restore vmadot registers
	vmv.v.i v16, 0
	vmv.v.i v17, 0
	vmv.v.i v18, 0
	vmv.v.i v19, 0

	vsetvli x7, zero, e8, m1

	# Increment pointers to next 4x8 submatrices
	add ap, ap, 32    	# Move A pointer to next chunk
	add bp, bp, t1    	# Move B pointer to next chunk (We must go to the next column here)

	addi kt, kt, 2    	# Increment K loop counter
	j k_loop_choose


k_loop4:

	# Load A and B (assume A is row-major and B is row-major)
	vle8.v v2, (ap)   	# Load 4x8 elements from A
	vle8.v v3, (bp)   	# Load 8x4 elements from B
	add ap, ap, 32    	# Move A pointer to next chunk
	add bp, bp, t1    	# Move B pointer to next chunk (We must go to the next column here)
    vle8.v v4, (ap)   	# Load 4x8 elements from A
	vle8.v v5, (bp)   	# Load 8x4 elements from B
	add ap, ap, 32    	# Move A pointer to next chunk
	add bp, bp, t1    	# Move B pointer to next chunk (We must go to the next column here)
    vle8.v v6, (ap)   	# Load 4x8 elements from A
	vle8.v v7, (bp)   	# Load 8x4 elements from B
	add ap, ap, 32    	# Move A pointer to next chunk
	add bp, bp, t1    	# Move B pointer to next chunk (We must go to the next column here)
    vle8.v v8, (ap)   	# Load 4x8 elements from A
	vle8.v v9, (bp)   	# Load 8x4 elements from B

	# Perform matrix multiplication
	vmadot v16, v2, v3
	vmadot v18, v4, v5
	vmadot v20, v6, v7
	vmadot v22, v8, v9

	# Store result in v0, v1
	vsetvli x7, zero, e32, m1
	vadd.vv v0, v0, v16
	vadd.vv v1, v1, v17  
	vadd.vv v0, v0, v18
	vadd.vv v1, v1, v19  
	vadd.vv v0, v0, v20
	vadd.vv v1, v1, v21  
	vadd.vv v0, v0, v22
	vadd.vv v1, v1, v23

	# Restore vmadot registers
	vmv.v.i v16, 0
	vmv.v.i v17, 0
	vmv.v.i v18, 0
	vmv.v.i v19, 0
	vmv.v.i v20, 0
	vmv.v.i v21, 0
	vmv.v.i v22, 0
	vmv.v.i v23, 0

	vsetvli x7, zero, e8, m1

	# Increment pointers to next 4x8 submatrices
	add ap, ap, 32    	# Move A pointer to next chunk
	add bp, bp, t1    	# Move B pointer to next chunk (We must go to the next column here)

	addi kt, kt, 4    	# Increment K loop counter
	j k_loop_choose


k_loop7:

	# Load A and B (assume A is row-major and B is row-major)
	vle8.v v2, (ap)   	# Load 4x8 elements from A
	vle8.v v3, (bp)   	# Load 8x4 elements from B
	add ap, ap, 32    	# Move A pointer to next chunk
	add bp, bp, t1    	# Move B pointer to next chunk (We must go to the next column here)
    vle8.v v4, (ap)   	# Load 4x8 elements from A
	vle8.v v5, (bp)   	# Load 8x4 elements from B
	add ap, ap, 32    	# Move A pointer to next chunk
	add bp, bp, t1    	# Move B pointer to next chunk (We must go to the next column here)
    vle8.v v6, (ap)   	# Load 4x8 elements from A
	vle8.v v7, (bp)   	# Load 8x4 elements from B
	add ap, ap, 32    	# Move A pointer to next chunk
	add bp, bp, t1    	# Move B pointer to next chunk (We must go to the next column here)
    vle8.v v8, (ap)   	# Load 4x8 elements from A
	vle8.v v9, (bp)   	# Load 8x4 elements from B
    add ap, ap, 32    	# Move A pointer to next chunk
	add bp, bp, t1    	# Move B pointer to next chunk (We must go to the next column here)
	vle8.v v10, (ap)   	# Load 4x8 elements from A
	vle8.v v11, (bp)   	# Load 8x4 elements from B
	add ap, ap, 32    	# Move A pointer to next chunk
	add bp, bp, t1    	# Move B pointer to next chunk (We must go to the next column here)
    vle8.v v12, (ap)   	# Load 4x8 elements from A
	vle8.v v13, (bp)   	# Load 8x4 elements from B
	add ap, ap, 32    	# Move A pointer to next chunk
	add bp, bp, t1    	# Move B pointer to next chunk (We must go to the next column here)
    vle8.v v14, (ap)   	# Load 4x8 elements from A
	vle8.v v15, (bp)   	# Load 8x4 elements from B

	# Perform matrix multiplication
	vmadot v16, v2, v3
	vmadot v18, v4, v5
	vmadot v20, v6, v7
	vmadot v22, v8, v9
	vmadot v24, v10, v11
	vmadot v26, v12, v13
	vmadot v28, v14, v15

	# Store result in v0, v1
	vsetvli x7, zero, e32, m1
	vadd.vv v0, v0, v16
	vadd.vv v1, v1, v17  
	vadd.vv v0, v0, v18
	vadd.vv v1, v1, v19  
	vadd.vv v0, v0, v20
	vadd.vv v1, v1, v21  
	vadd.vv v0, v0, v22
	vadd.vv v1, v1, v23  
	vadd.vv v0, v0, v24
	vadd.vv v1, v1, v25  
	vadd.vv v0, v0, v26
	vadd.vv v1, v1, v27  
	vadd.vv v0, v0, v28
	vadd.vv v1, v1, v29  

	# Restore vmadot registers
	vmv.v.i v16, 0
	vmv.v.i v17, 0
	vmv.v.i v18, 0
	vmv.v.i v19, 0
	vmv.v.i v20, 0
	vmv.v.i v21, 0
	vmv.v.i v22, 0
	vmv.v.i v23, 0
	vmv.v.i v24, 0
	vmv.v.i v25, 0
	vmv.v.i v26, 0
	vmv.v.i v27, 0
	vmv.v.i v28, 0
	vmv.v.i v29, 0

	vsetvli x7, zero, e8, m1

	# Increment pointers to next 4x8 submatrices
	add ap, ap, 32    	# Move A pointer to next chunk
	add bp, bp, t1    	# Move B pointer to next chunk (We must go to the next column here)

	addi kt, kt, 7    	# Increment K loop counter
	j k_loop_choose


k_done:

    # Restore A and B pointers correctly for next row/column block
	mv ap, akp
	mv bp, bkp
	add bp, bp, 32      # Move B to next 8x4 block

	# Set vector length for storing int32 (result)
	vsetvli x7, zero, e32, m1

	# Store results into C (matrix)
	vse32.v v0, (cp)  
	add cp, cp, 32   	# Update C pointer based on vector length set with vsetvli
	vse32.v v1, (cp)
    add cp, cp, 32      # Move C to accept next 4x4 block

	# Increment column counter for N loop
	addi nt, nt, 1
    
	j n_loop

n_done:

    # Move A to next row block and reset B
    li t1, 32 
    mul t1, t1, k   # To shift A to next row, multiply by K
	add ap, ap, t1 
	mv bp, borg     # Restore B address to initial value

	# Increment row counter for M loop
	addi mt, mt, 1

    li nt, 0
	j m_loop

m_done:
	ret

