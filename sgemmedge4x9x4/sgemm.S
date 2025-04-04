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
#define k_rem a4
#define bp a5
#define borg a6
#define cp a7
#define mt t0
#define kt t3
#define nt t4
#define akp t5
#define bkp t6

#define ckp s0
#define cacc s1

sgemm_nn:

     addi sp, sp, -24   # Allocate stack space
     sd s0, 0(sp)   	# Save s0
     sd s1, 8(sp)   	# Save s1


    mv borg, bp         # Store initial b address
    li t0, 4            # Load auxiliar value to find iteration values. N / 4 & M / 4 (due to 4x8x4 MAC)
    div n, n, t0
    div m, m, t0
    li t0, 8            # Load auxiliar value to find iteration values. K / 8 (due to 4x8x4 MAC)
    rem k_rem, k, t0
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

k_loop:

	bge kt, k, k_done
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
	j k_loop

cleanup_k_loop_start:
	# Set vector length to handle K_rem elements (not needed anymore since we're doing scalar ops)
	li kt, 0  # Start at the first remaining K index
	mv bkp, bp
    mv ckp, cp
    slli n, n, 2
    # slli m, m, 2
    slli k_rem, k_rem, 2

cleanup_k_loop:
	# If kt == K_rem, we're done
    # esto deberia ser 4 * k_rem
	bge kt, k_rem, cleanup_done

	li mt, 0

cleanup_m_loop:
	# If m == M, move to next kr
	bge mt, m, cleanup_next_k

	li nt, 0

cleanup_n_loop:
	# If n == N, move to next row
	bge nt, n, cleanup_next_m
                                     #TODO aqui no se puede usar t0 porque perderiamos mt
	lb t1, 0(ap)
	lb t2, 0(bp)

	mul cacc, t1, t2
    # add a v0 o v1 dependiendo de por donde vamos;
	sw cacc, 0(ckp)

	add bp, bp, 1   # *k_rem 
    add ckp, ckp, 4 # esto ahora depende de k_rem, para los saltos de columna (4*k_rem)
	addi nt, nt, 1  # *k_rem ?
	j cleanup_n_loop

cleanup_next_m:
	# Move to the next row in A
	add ap, ap, 1
    mv bp, bkp
	addi mt, mt, 1
	j cleanup_m_loop

cleanup_next_k:
	addi kt, kt, 1
	j cleanup_k_loop

cleanup_done:
    mv ckp, cp
	mv k_rem, zero

k_done:
    bgt k_rem, zero, cleanup_k_loop_start
    # Restore A and B pointers correctly for next row/column block
	mv ap, akp
	mv bp, bkp
	add bp, bp, 32      # Move B to next 8x4 block

	# Set vector length for storing int32 (result)
	vsetvli x7, zero, e32, m1

	# Store results into C (matrix)
    vle32.v v2, (cp)    # Load whatever was on C before, then add it to the new values
    vadd.vv v0, v0, v2
	vse32.v v0, (cp)  
	add cp, cp, 32   	# Update C pointer based on vector length set with vsetvli
    vle32.v v3, (cp)    # Load whatever was on C before, then add it to the new values
    vadd.vv v1, v1, v3
	vse32.v v1, (cp)
    add cp, cp, 32      # Move C to accept next 4x4 block

	# Increment column counter for N loop
	addi nt, nt, 1
    srli n, n, 2
    # srli m, m, 2
    # srli k_rem, k_rem, 2
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

     ld s0, 0(sp)   	# Restore s0
     ld s1, 8(sp)   	# Restore s1
     addi sp, sp, 24	# Deallocate stack space

	ret

