function run() {
    echo -e "\nPerformance de tiny_md con N=$1:\n" >> results.txt
    make -B CFLAGS="-O3 -DN=$1" tiny_md
    perf stat ./tiny_md >> results.txt 2>> results.txt
}

echo -e "\nBenchmark $(date +%s):" >> results.txt

run 256
run 500
run 864
run 1372
run 2048
