#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate mlfold

# Create a timestamped output directory
timestamp=$(date +%Y%m%d_%H%M%S)
comparison_dir="outputs/comparison_${timestamp}"
mkdir -p $comparison_dir

# Create output directories
folder_with_pdbs="inputs/PDB_monomers/pdbs/"
output_dir="${comparison_dir}/torchscript"
original_dir="${comparison_dir}/original"
mkdir -p $output_dir/seqs
mkdir -p $original_dir/seqs

# First generate the parsed pdbs file
path_for_parsed_chains=$comparison_dir"/parsed_pdbs.jsonl"
python helper_scripts/parse_multiple_chains.py --input_path=$folder_with_pdbs --output_path=$path_for_parsed_chains

# Run with original implementation
echo "Running with original implementation..."
python protein_mpnn_run.py \
    --jsonl_path $path_for_parsed_chains \
    --out_folder $original_dir \
    --num_seq_per_target 2 \
    --sampling_temp "0.1" \
    --seed 37 \
    --batch_size 1 > $comparison_dir/original_output.log 2>&1

# Run with torchscript implementation
echo "Running with torchscript implementation..."
python protein_mpnn_run.py \
    --jsonl_path $path_for_parsed_chains \
    --out_folder $output_dir \
    --num_seq_per_target 2 \
    --sampling_temp "0.1" \
    --seed 37 \
    --batch_size 1 \
    --use_torchscript > $comparison_dir/torchscript_output.log 2>&1

# Compare sequence outputs
echo "Comparing sequence outputs..."
echo "Original sequences:"
cat $original_dir/seqs/*.fa
echo -e "\nTorchscript sequences:"
cat $output_dir/seqs/*.fa

# Save comparison results
{
    echo "Comparison Results (${timestamp})"
    echo "================================"
    echo -e "\nOriginal sequences:"
    cat $original_dir/seqs/*.fa
    echo -e "\nTorchscript sequences:"
    cat $output_dir/seqs/*.fa
    echo -e "\nOriginal scores:"
    grep "score=" $comparison_dir/original_output.log | cut -d',' -f2-3
    echo -e "\nTorchscript scores:"
    grep "score=" $comparison_dir/torchscript_output.log | cut -d',' -f2-3
} > $comparison_dir/comparison_results.txt

echo -e "\nComparison results saved to: $comparison_dir/comparison_results.txt" 