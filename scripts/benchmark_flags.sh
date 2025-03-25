function run() {
    N=500
    echo -e "\nPerformance de tiny_md con flags=$1:\n" >> results.txt
    make -B CFLAGS="$1 -DN=$N" tiny_md
    perf stat ./tiny_md >> results.txt 2>> results.txt
}

echo -e "\nBenchmark $(date +%s):" >> results.txt

run "-O0"
run "-O1"
run "-O2"
run "-O3"
