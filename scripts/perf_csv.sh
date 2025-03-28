if [ $# -lt 1 ]; then
	echo "Falta el nombre de un ejecutable"
	exit 1
fi

if ! [ -e $1 ]; then
	echo "No existe el archivo $1"
	exit 1
fi

ESTADISTICAS=task-clock,cpu-clock,instructions,cache-references,cache-misses,branches,branch-misses 
ARCHIVO=run_$1_$(date +%s).csv
COMANDO=$(realpath "$1")

echo "Ejecutando $1 con perf..."
perf stat -o $ARCHIVO -e $ESTADISTICAS -x ";" $COMANDO
echo "Resultados generados en $ARCHIVO"
