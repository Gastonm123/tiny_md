function run() {
    echo -e "\nPerformance de tiny_md con N=$1 particulas:\n" >> results.txt
    make -B CFLAGS="-O3 -DN=$1 -ffast-math -DMAN_UNROLL" CC=icx tiny_md
    /usr/bin/time ./tiny_md >> /dev/null 2>> results.txt
}

echo -e "\nBenchmark $(date +%s):" >> results.txt

run 4    # m = 1
run 32   # m = 2
run 108  # m = 3
run 256  # m = 4
run 500  # m = 5
run 864  # m = 6
run 1372 # m = 7
run 2048 # m = 8
run 2916
run 4000
