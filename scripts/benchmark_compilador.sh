function run() {
    make -B CFLAGS="$1 -DN=500" CC="$2" tiny_md
    RES=$(perf stat -x ", " -e fp_ret_sse_avx_ops.all,instructions,task-clock ./tiny_md 2>&1 >/dev/null | awk -F', ' '{print $1}' | paste -sd "," -)
    echo "$2,$1,$RES" >> results.txt
}

echo -e "\nBenchmark $(date +%s):" >> results.txt

declare -a flags=("-O0" "-O1" "-O2" "-O3 -ffast-math")
declare -a compi=("gcc" "clang" "icx")

for (( i = 0; i<${#flags[@]}; i++)); do
    for (( j = 0; j<${#compi[@]}; j++)); do
        #echo "${flags[$i]}" "${compi[$j]}"
        run "${flags[$i]}" "${compi[$j]}"
    done
done