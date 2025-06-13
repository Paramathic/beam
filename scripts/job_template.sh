#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=rrg-mmehride
#SBATCH --job-name=learn_mask # Base name, will be overridden by submit_jobs.sh


ARG_MODEL_NAME="${1:-'meta-llama/Llama-3.2-1B'}"
ARG_SPARSITY_RATIO="${2:-0.5}"
ARG_COPY_DATA="${3:-false}"
ARG_SPARSITY_TYPE="${4:-'unstructured'}"
ARG_PRUNING_METHOD="${5:-'wanda'}"
ARG_BEAM="${6:-false}"
ARG_BEAM_ONLINE_TUNE="${7:-true}"
ARG_BEAM_BLOCK_GRANULARITY="${8:-1}"
ARG_LORA_RANK="${9:-0}"
ARG_SLIM_LORA="${10:-false}"
ARG_QUANTIZE_WEIGHT="${11:-false}"
ARG_BEAM_NUM_SAMPLES="${12:-128}"
ARG_SLIM_QUANT="${13:-false}"
ARG_OPTIMIZER="${14:-'adam'}"
ARG_LAYER_WISE_OPTIMIZATION="${15:-false}"
ARG_NUM_BEAM_EPOCHS="${16:-32}"
ARG_TILED_WEIGHT_QUANTIZATION="${17:-true}"

SCRIPT_TO_RUN=scripts/run_beam_args.sh

echo "Starting SLURM job $SLURM_JOB_ID for job name $SLURM_JOB_NAME"
echo "Using SLURM temporary directory: $SLURM_TMPDIR"
echo "Received arguments:"
echo "  MODEL_NAME: $ARG_MODEL_NAME"
echo "  SPARSITY_RATIO: $ARG_SPARSITY_RATIO"
echo "  COPY_DATA: $ARG_COPY_DATA"
echo "  SPARSITY_TYPE: $ARG_SPARSITY_TYPE"
echo "  PRUNING_METHOD: $ARG_PRUNING_METHOD"
echo "  BEAM: $ARG_BEAM"
echo "  BEAM_ONLINE_TUNE: $ARG_BEAM_ONLINE_TUNE"
echo "  BEAM_BLOCK_GRANULARITY: $ARG_BEAM_BLOCK_GRANULARITY"
echo "  LORA_RANK: $ARG_LORA_RANK"
echo "  SLIM_LORA: $ARG_SLIM_LORA"
echo "  QUANTIZE_WEIGHT: $ARG_QUANTIZE_WEIGHT"
echo "  BEAM_NUM_SAMPLES: $ARG_BEAM_NUM_SAMPLES"
echo "  SLIM_QUANT: $ARG_SLIM_QUANT"
echo "  OPTIMIZER: $ARG_OPTIMIZER"
echo "  LAYER_WISE_OPTIMIZATION: $ARG_LAYER_WISE_OPTIMIZATION"
echo "  NUM_BEAM_EPOCHS: $ARG_NUM_BEAM_EPOCHS"
echo "  TILED_WEIGHT_QUANTIZATION: $ARG_TILED_WEIGHT_QUANTIZATION"

module load apptainer 
export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="$SLURM_TMPDIR/data" 
export MASTER_PORT=29501
export OMP_NUM_THREADS=12

mkdir -p "$HF_HOME"

USERNAME=$(whoami)


# --- Data and Container Preparation ---
if [ "$ARG_COPY_DATA" = true ]; then
    echo "Copying data to SLURM_TMPDIR..."
    DATA_DIR_SRC="/home/${USERNAME}/projects/def-mmehride/${USERNAME}/data"
    DATA_DIR_TMP="$SLURM_TMPDIR/data"
    cp -r "$DATA_DIR_SRC" "$SLURM_TMPDIR/"
    echo "Data copied to $DATA_DIR_TMP"
else
    echo "Skipping data copy as per user request."
    DATA_DIR_TMP="/home/${USERNAME}/projects/def-mmehride/${USERNAME}/data" # Use the original data directory
fi

echo "Preparing container..."
rm -rf $SLURM_TMPDIR/torch-one-shot.sif;
mkdir ${SLURM_TMPDIR}/torch-one-shot.sif;
tar -xf /home/${USERNAME}/projects/def-mmehride/${USERNAME}/torch-one-shot.tar -C $SLURM_TMPDIR;
mkdir ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki;
mkdir ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki/tls;
mkdir ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki/tls/certs;
cp /etc/ssl/certs/ca-bundle.crt ${SLURM_TMPDIR}/torch-one-shot.sif/etc/pki/tls/certs/ca-bundle.crt;

# --- Execute inside Singularity ---
echo "Executing run_experiment.sh inside Singularity..."
singularity exec \
    --bind $PWD:/home/${USERNAME} \
    --bind $SLURM_TMPDIR:/tmp \
    --nv \
    ${SLURM_TMPDIR}/torch-one-shot.sif \
    mkdir -p /home/${USERNAME}/data
    
singularity exec \
    --bind $PWD:/home/${USERNAME} \
    --bind $SLURM_TMPDIR:/tmp \
    --bind $DATA_DIR_TMP:/home/${USERNAME}/data \
    --nv ${SLURM_TMPDIR}/torch-one-shot.sif \
    bash "${SCRIPT_TO_RUN}" "${ARG_MODEL_NAME}" "${ARG_SPARSITY_RATIO}" "${ARG_COPY_DATA}" "${ARG_SPARSITY_TYPE}" \
        "${ARG_PRUNING_METHOD}" "${ARG_BEAM}" "${ARG_BEAM_ONLINE_TUNE}" "${ARG_BEAM_BLOCK_GRANULARITY}" "${ARG_LORA_RANK}" \
        "${ARG_SLIM_LORA}" "${ARG_QUANTIZE_WEIGHT}" "${ARG_BEAM_NUM_SAMPLES}" "${ARG_SLIM_QUANT}" "${ARG_OPTIMIZER}" \
        "${ARG_LAYER_WISE_OPTIMIZATION}" "${ARG_NUM_BEAM_EPOCHS}" "${ARG_TILED_WEIGHT_QUANTIZATION}"

echo "Singularity execution finished successfully."
echo "SLURM Job $SLURM_JOB_ID finished."