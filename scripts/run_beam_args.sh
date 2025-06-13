export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="data"

export HF_TOKEN="HF_TOKEN"
HF_TOKEN_ARG="--hf_token HF_TOKEN"
export WANDB_API_KEY=WANDB_TOKEN

MODEL_NAME="${1:-'meta-llama/Llama-3.2-1B'}"
SPARSITY_RATIO="${2:-0.5}"
COPY_DATA="${3:-false}"
STRUCTURE="${4:-'unstructured'}"
METHOD="${5:-'wanda'}"
BEAM="${6:-false}"
BEAM_ONLINE_TUNE="${7:-true}"
BEAM_BLOCK_GRANULARITY="${8:-1}"
LORA_RANK="${9:-0}"
SLIM_LORA="${10:-false}"
QUANTIZE_WEIGHT="${11:-false}"
BEAM_NUM_SAMPLES="${12:-128}"
SLIM_QUANT="${13:-false}"
OPTIMIZER="${14:-'adam'}"
LAYER_WISE_OPTIMIZATION="${15:-false}"
NUM_EPOCHS="${16:-32}"
TILED_WEIGHT_QUANTIZATION="${17:-true}"

if [ "$BEAM" == "true" ]; then
    BEAM="--beam"
else
    BEAM=""
fi

if [ "$BEAM_ONLINE_TUNE" == "true" ]; then
    BEAM_ONLINE_TUNE="--beam_online_tune"
else
    BEAM_ONLINE_TUNE=""
fi

if [ "$SLIM_LORA" == "true" ]; then
    SLIM_LORA="--slim_lora"
else
    SLIM_LORA=""
fi

if [ "$QUANTIZE_WEIGHT" == "true" ]; then
    QUANTIZE_WEIGHT="--quantize_weight"
else
    QUANTIZE_WEIGHT=""
fi

if [ "$SLIM_QUANT" == "true" ]; then
    SLIM_QUANT="--slim_quant"
else
    SLIM_QUANT=""
fi

if [ "$TILED_WEIGHT_QUANTIZATION" == "true" ]; then
    TILED_WEIGHT_QUANTIZATION="--tiled_weight_quantization"
else
    TILED_WEIGHT_QUANTIZATION=""
fi

NUM_CALIBRATION_SAMPLES=128
# LOCAL_FILES_ONLY='--local_files_only'
SHIFT_ZERO_METRICS='--shift_zero_metrics'
EVAL_DATASET='wikitext2'
BITWIDTH=4
# QUANTIZE_INPUT='--quantize_input'
INPUT_BITWIDTH=8
INPUT_GROUP_SIZE=-1
EVAL_BATCH_SIZE=1
SEPARATE_LORA='--separate_lora'
TEST_LMHARNESS='--test_lmharness'
# FINE_TUNE='--fine_tune'
EVALUATE_PERPLEXITY='--evaluate_perplexity'
# PRUNE_LORA="--prune_lora"
# QUANTIZE_LORA="--quantize_lora"
LORA_TILE_SIZE=128
WEIGHT_TILE_SIZE=32
PAD_LORA='--pad_lora'
CALIBRATION_DATASET="c4"
WANDB="--wandb"

RESULTS_FILE_NAME=results/$METHOD.csv

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model $MODEL_NAME \
    --prune_method $METHOD \
    --sparsity_ratio $SPARSITY_RATIO \
    --sparsity_type $STRUCTURE \
    --lora_rank $LORA_RANK \
    $SLIM_LORA \
    --eval_dataset $EVAL_DATASET \
    $SHIFT_ZERO_METRICS \
    $QUANTIZE_WEIGHT \
    --bitwidth $BITWIDTH \
    $SLIM_QUANT \
    --eval_batch_size $EVAL_BATCH_SIZE \
    $SEPARATE_LORA \
    $TEST_LMHARNESS \
    $FINE_TUNE \
    $EVALUATE_PERPLEXITY \
    $LOCAL_FILES_ONLY \
    $QUANTIZE_INPUT \
    --input_bitwidth $INPUT_BITWIDTH \
    --input_group_size $INPUT_GROUP_SIZE \
    --compression_nsamples $NUM_CALIBRATION_SAMPLES \
    --beam_nsamples $BEAM_NUM_SAMPLES \
    --optimizer $OPTIMIZER \
    $TILED_INPUT_QUANTIZATION \
    $PRUNE_LORA \
    $QUANTIZE_LORA \
    --lora_tile_size $LORA_TILE_SIZE \
    $TILED_WEIGHT_QUANTIZATION \
    --weight_tile_size $WEIGHT_TILE_SIZE \
    $HF_TOKEN_ARG \
    $BEAM \
    --num_epochs $NUM_EPOCHS \
    --beam_block_granularity $BEAM_BLOCK_GRANULARITY \
    --output_csv_path $RESULTS_FILE_NAME \
    $PAD_LORA \
    --calibration_dataset $CALIBRATION_DATASET \
    $WANDB \
    $BEAM_ONLINE_TUNE