#!/bin/bash

# --- Configuration ---
# Define the ranges for your hyperparameters
SPARSITY_RATIO=(0.5)
COPY_DATA=false
SPARSITY_TYPE_VALUES=("2:4") # "unstructured")
PRUNING_METHOD_VALUES=("wanda")
MODEL_NAME=gemma3
BEAM=true
BEAM_ONLINE_TUNE=true
BEAM_BLOCK_GRANULARITY_VALUES=(4)
LORA_RANK=0.1
SLIM_LORA=true
QUANTIZE_WEIGHT_VALUES=(false true)
BEAM_NUM_SAMPLES=128
SLIM_QUANT=true
OPTIMIZER="adam"
LAYER_WISE_OPTIMIZATION=false
NUM_BEAM_EPOCHS=32

if [ $SLIM_QUANT == true ]
then
    TILED_WEIGHT_QUANTIZATION=false
else
    TILED_WEIGHT_QUANTIZATION=true
fi


NGPUS_PER_NODE=1
NTASKS_PER_NODE=$((8 * NGPUS_PER_NODE))
MEM=$((45 * NGPUS_PER_NODE))
GPU_TYPE="v100l:"
TIME="6:00:00"

if [ $MODEL_NAME == 'llama2' ]
then
    MODEL_PREFIX=meta-llama/Llama-2-
    MODEL_POSTFIX=-hf
    MODEL_SIZE_LIST='7b 13b'
elif [ $MODEL_NAME == 'opt' ]
then   
    MODEL_PREFIX=facebook/opt-
    MODEL_POSTFIX=''
    MODEL_SIZE_LIST='125m 350m 1.3b 2.7b 6.7b 13b'
elif [ $MODEL_NAME == 'llama3.2' ]
then
    MODEL_PREFIX=meta-llama/Llama-3.2-
    MODEL_SIZE_LIST='1B 3B'
    MODEL_POSTFIX=''
elif [ $MODEL_NAME == 'llama3.1' ]
then
    MODEL_PREFIX=meta-llama/Llama-3.1-
    MODEL_SIZE_LIST='8B'
    MODEL_POSTFIX=''
elif [ $MODEL_NAME == 'gemma3' ]
then
    MODEL_PREFIX=google/gemma-3-
    MODEL_SIZE_LIST='1b 4b 12b'
    MODEL_POSTFIX='-pt'
fi

SLURM_SCRIPT="scripts/job_template.sh"
EXPERIMENT_SCRIPT="scripts/run_learnable_mask.sh"

echo "Starting job submission loop..."

# Loop through all combinations
job_count=0
for target_sparsity in "${SPARSITY_RATIO[@]}"; do
    for MODEL_SIZE in $MODEL_SIZE_LIST; do
        for BEAM_BLOCK_GRANULARITY in "${BEAM_BLOCK_GRANULARITY_VALUES[@]}"; do
            for QUANTIZE_WEIGHT in "${QUANTIZE_WEIGHT_VALUES[@]}"; do
                for PRUNING_METHOD in "${PRUNING_METHOD_VALUES[@]}"; do
                    for SPARSITY_TYPE in "${SPARSITY_TYPE_VALUES[@]}"; do
                        JOB_NAME="beam_${MODEL_NAME}_${MODEL_SIZE}_${SPARSITY_RATIO}_BEAM_${BEAM}_BEAM_ONLINE_TUNE_${BEAM_ONLINE_TUNE}_BEAM_BLOCK_GRANULARITY_${BEAM_BLOCK_GRANULARITY}_LORA_RANK_${LORA_RATIO}_SLIM_LORA_${SLIM_LORA}"
                        JOB_NAME=$(echo "$JOB_NAME" | sed 's/e-/em/' | sed 's/[^A-Za-z0-9._-]/_/g')

                        echo "--------------------------------------------------"
                        echo "Submitting job #$((job_count + 1)): ${JOB_NAME}"

                        sbatch --account=rrg-mmehride \
                            --job-name="${GPU_TYPE}${JOB_NAME}" \
                            --gpus-per-node=${NGPUS_PER_NODE} \
                            --ntasks-per-node=${NTASKS_PER_NODE} \
                            --mem=${MEM}G \
                            --time=${TIME} \
                            "${SLURM_SCRIPT}" \
                            "${MODEL_PREFIX}${MODEL_SIZE}${MODEL_POSTFIX}" \
                            "${SPARSITY_RATIO}" \
                            "${COPY_DATA}" \
                            "${SPARSITY_TYPE}" \
                            "${PRUNING_METHOD}" \
                            "${BEAM}" \
                            "${BEAM_ONLINE_TUNE}" \
                            "${BEAM_BLOCK_GRANULARITY}" \
                            "${LORA_RANK}" \
                            "${SLIM_LORA}" \
                            "${QUANTIZE_WEIGHT}" \
                            "${BEAM_NUM_SAMPLES}" \
                            "${SLIM_QUANT}" \
                            "${OPTIMIZER}" \
                            "${LAYER_WISE_OPTIMIZATION}" \
                            "${NUM_BEAM_EPOCHS}" \
                            "${TILED_WEIGHT_QUANTIZATION}"

                        ((job_count++))
                    done
                done
            done
        done
    done
done

echo "--------------------------------------------------"
echo "Finished submitting ${job_count} jobs."