export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="data"
# export HF_DATASETS_OFFLINE="1"
# export HF_HUB_OFFLINE="1"

HF_TOKEN="--hf_token HF_TOKEN"
export WANDB_API_KEY=WANDB_TOKEN


for MODEL_NAME in llama3.2 #gemma3
do
    if [ $MODEL_NAME == 'llama2' ]
    then
        MODEL_PREFIX=meta-llama/Llama-2-
        MODEL_POSTFIX=-hf
        MODEL_SIZE_LIST="7b 13b"
    elif [ $MODEL_NAME == 'opt' ]
    then   
        MODEL_PREFIX=facebook/opt-
        MODEL_POSTFIX=""
        MODEL_SIZE_LIST="125m 350m 1.3b 2.7b 6.7b 13b"
    elif [ $MODEL_NAME == 'llama3.2' ]
    then
        MODEL_PREFIX=meta-llama/Llama-3.2-
        MODEL_SIZE_LIST="1B 3B"
        MODEL_POSTFIX=""
    elif [ $MODEL_NAME == 'llama3.1' ]
    then
        MODEL_PREFIX=meta-llama/Llama-3.1-
        MODEL_SIZE_LIST="8B"
        MODEL_POSTFIX=""
    elif [ $MODEL_NAME == 'gemma3' ]
    then
        MODEL_PREFIX=google/gemma-3-
        MODEL_SIZE_LIST="1b 4b 12b"
        MODEL_POSTFIX="-pt"
    fi
    


    for MODEL_SIZE in $MODEL_SIZE_LIST
    do
        for STRUCTURE in 2:4 #unstructured
        do
            for METHOD in wanda #sparsegpt
            do
                for LORA_RANK in 0
                do
                    for SLIM_LORA in '' #'--slim_lora'
                    do
                        for NUM_CALIBRATION_SAMPLES in 128
                        do
                            for QUANTIZE_WEIGHT in '--quantize_weight'
                            do
                                for BEAM_BLOCK_GRANULARITY in 1 #2 4
                                do
                                    LOCAL_FILES_ONLY='--local_files_only'
                                    SPARSITY_RATIO=0.5
                                    SHIFT_ZERO_METRICS='--shift_zero_metrics'
                                    EVAL_DATASET='wikitext2'
                                    BITWIDTH=4
                                    # QUANTIZE_INPUT='--quantize_input'
                                    INPUT_BITWIDTH=8
                                    INPUT_GROUP_SIZE=-1
                                    # SLIM_QUANT='--slim_quant'
                                    EVAL_BATCH_SIZE=1
                                    SEPARATE_LORA='--separate_lora'
                                    TEST_LMHARNESS='--test_lmharness'
    #                                FINE_TUNE='--fine_tune'
                                    EVALUATE_PERPLEXITY='--evaluate_perplexity'
                                    OPTIMIZER="adam"
    #                                PRUNE_LORA="--prune_lora"
                                    # QUANTIZE_LORA="--quantize_lora"
                                    LORA_TILE_SIZE=256
                                    TILED_WEIGHT_QUANTIZATION="--tiled_weight_quantization"
                                    WEIGHT_TILE_SIZE=32
                                    BEAM="--beam"
                                    BEAM_NUM_CALIBRATION_SAMPLES=128
                                    NUM_EPOCHS=16
                                    PAD_LORA='--pad_lora'
                                    CALIBRATION_DATASET="c4"
                                    # WANDB="--wandb"
                                    BEAM_ONLINE_TUNE="--beam_online_tune"

                                    RESULTS_FILE_NAME=results/$METHOD.csv

                                    CUDA_VISIBLE_DEVICES=0 python main.py \
                                        --model ${MODEL_PREFIX}${MODEL_SIZE}${MODEL_POSTFIX} \
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
                                        --beam_nsamples $BEAM_NUM_CALIBRATION_SAMPLES \
                                        --optimizer $OPTIMIZER \
                                        $TILED_INPUT_QUANTIZATION \
                                        $PRUNE_LORA \
                                        $QUANTIZE_LORA \
                                        --lora_tile_size $LORA_TILE_SIZE \
                                        $TILED_WEIGHT_QUANTIZATION \
                                        --weight_tile_size $WEIGHT_TILE_SIZE \
                                        $HF_TOKEN \
                                        $BEAM \
                                        --num_epochs $NUM_EPOCHS \
                                        --beam_block_granularity $BEAM_BLOCK_GRANULARITY \
                                        --output_csv_path $RESULTS_FILE_NAME \
                                        $PAD_LORA \
                                        --calibration_dataset $CALIBRATION_DATASET \
                                        $WANDB \
                                        $BEAM_ONLINE_TUNE
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
