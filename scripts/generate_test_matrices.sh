#!/bin/bash
# Generate synthetic sparse matrices for SpMV benchmark
# Matrices simulate avgNnz ~ 10 (same as real test cases)

set -e

OUTPUT_DIR=${1:-"./matrices"}
mkdir -p $OUTPUT_DIR

echo "=== Generating Synthetic Sparse Matrices ==="
echo "Output: $OUTPUT_DIR"
echo ""

# Matrix specs: rows, avg_nnz_per_row
MATRICES=(
    "p0_A:100000:10"
    "p1_A:200000:10"
    "p2_A:500000:10"
    "p3_A:1000000:10"
    "p4_A:100000:5"
    "p5_A:100000:20"
    "p6_A:100000:30"
    "p7_A:100000:50"
    "p8_A:100000:100"
    "p9_A:50000:8"
)

for spec in "${MATRICES[@]}"; do
    name=$(echo $spec | cut -d: -f1)
    rows=$(echo $spec | cut -d: -f2)
    avg_nnz=$(echo $spec | cut -d: -f3)

    cols=$rows
    nnz=$((rows * avg_nnz))

    output_file="$OUTPUT_DIR/${name}.mtx"

    echo "Generating $name: $rows x $cols, avgNnz=$avg_nnz, total_nnz=$nnz"

    # Write MTX header
    echo "%%MatrixMarket matrix coordinate real general" > $output_file
    echo "$rows $cols $nnz" >> $output_file

    # Generate random sparse data using shell
    # For each row, generate avg_nnz entries with random columns
    awk -v rows=$rows -v cols=$cols -v avg_nnz=$avg_nnz 'BEGIN {
        srand();
        for (r = 1; r <= rows; r++) {
            for (i = 0; i < avg_nnz; i++) {
                c = int(rand() * cols) + 1;
                val = rand() * 2 - 1;  # Random value -1 to 1
                printf "%d %d %.15e\n", r, c, val;
            }
        }
    }' >> $output_file

    echo "  -> Done: $(wc -l < $output_file) lines"
done

echo ""
echo "=== Generation Complete ==="