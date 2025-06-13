export HF_DATASETS_TRUST_REMOTE_CODE="1"
export HF_HOME="data"

# python utils/download_data.py --model 'lmharness'
# python utils/download_data.py --model data

for MODEL in llama3_1
do
    python utils/download_data.py --local_cache --model $MODEL
    python utils/download_data.py --model $MODEL
done