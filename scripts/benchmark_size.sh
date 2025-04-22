FLAGS="-O3 -ffast-math -march=native -flto -funroll-loops -ftree-vectorize"

function run() {
    make -B CFLAGS="$FLAGS -DN=$1" CC=gcc tiny_md
    RES=$(perf stat -x ", " -e fp_ret_sse_avx_ops.all,instructions,task-clock ./tiny_md 2>&1 >/dev/null | awk -F', ' '{print $1}' | paste -sd "," -)
    echo "gcc,$1,$RES" >> results.txt
    make -B CFLAGS="$FLAGS -DN=$1" CC=clang tiny_md
    RES=$(perf stat -x ", " -e fp_ret_sse_avx_ops.all,instructions,task-clock ./tiny_md 2>&1 >/dev/null | awk -F', ' '{print $1}' | paste -sd "," -)
    echo "clang,$1,$RES" >> results.txt
    make -B CFLAGS="$FLAGS -DN=$1" CC=icx tiny_md
    RES=$(perf stat -x ", " -e fp_ret_sse_avx_ops.all,instructions,task-clock ./tiny_md 2>&1 >/dev/null | awk -F', ' '{print $1}' | paste -sd "," -)
    echo "icx,$1,$RES" >> results.txt
}

echo -e "\nBenchmark $(date +%s):" >> results.txt
echo "Flags: $FLAGS" >> results.txt
echo "Compiler,Size,Flop,Instructions,Time" >> results.txt

run 4    # m = 1
run 32   # m = 2
run 108  # m = 3
run 256  # m = 4
run 500  # m = 5
run 864  # m = 6
run 1372 # m = 7

