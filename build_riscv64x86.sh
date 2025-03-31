SRC=riscv64
ASM=$SRC/asm
COMM=common
BUILD_DIR=build_dir

# make directory
if [ -d "$BUILD_DIR" ]; then
    rm -rf $BUILD_DIR/*
else
    mkdir $BUILD_DIR
fi

# build common tools
riscv64-unknown-linux-gnu-g++ -O3 -c $COMM/table.cpp -o $BUILD_DIR/table.o
riscv64-unknown-linux-gnu-g++ -O3 -pthread -c $COMM/smtl.cpp -o $BUILD_DIR/smtl.o

# gen benchmark macro according to cpuid feature
SIMD_MACRO=" "
SIMD_OBJ=" "
SIMD_LIST="_IME_ _VECTOR_"
for SIMD in $SIMD_LIST;
do
    SIMD_MACRO="$SIMD_MACRO-D$SIMD "
    SIMD_OBJ="$SIMD_OBJ$BUILD_DIR/$SIMD.o "
    riscv64-unknown-linux-gnu-as -march=rv64gcv_zfh -c $ASM/$SIMD.S -o $BUILD_DIR/$SIMD.o
done

# compile cpufp
riscv64-unknown-linux-gnu-g++ -O3 -march=rv64gcv_zfh -I$COMM $SIMD_MACRO -c $SRC/cpufp.cpp -o $BUILD_DIR/cpufp.o
riscv64-unknown-linux-gnu-g++ -O3 -z noexecstack -pthread -o cpufp $BUILD_DIR/cpufp.o $BUILD_DIR/smtl.o $BUILD_DIR/table.o $SIMD_OBJ
