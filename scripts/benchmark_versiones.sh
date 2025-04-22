FLAGS="-O3 -ffast-math -flto -march=native -funroll-loops"

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

function benchmark_version() {
    git checkout $1
    echo -e "\nBenchmark $(date +%s): ($(git log -1 --pretty=format:%s $1))" >> results.txt
    echo "Compiler,Size,Flop,Instructions,Time" >> results.txt

    run 4    # m = 1
    run 32   # m = 2
    run 108  # m = 3
    run 256  # m = 4
    run 500  # m = 5
    run 864  # m = 6
    run 1372 # m = 7
}

benchmark_version 5a12119401adb7e17644ba3b09f556cabd3328a7
benchmark_version cbd68b9a16958fb8f7f2d23dcedcd543a03faaab
benchmark_version e28597ce83e89382d4e8b117da24371c1e9a100d