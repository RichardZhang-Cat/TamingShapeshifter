echo "Starting TAP vs defense test..." 


ROOT_DIR="."

# Switch to project root
cd $ROOT_DIR

export TAP_EVAL_PATH=$ROOT_DIR

# Configure CUDA device
export CUDA_VISIBLE_DEVICES=0


export OPENAI_API_BASE=""
export OPENAI_API_KEY=""


MAX_SAMPLES=50  
DATASET_PATH="./data/advbench_subset.csv"
RESULTS_FOLDER="results/$(date +%Y%m%d_%H%M%S)"

TARGET_MODEL="vicuna-13b"

echo "Test parameters:"
echo "  Max samples: $MAX_SAMPLES"
echo "  Dataset: $DATASET_PATH"
echo "  Results folder: $RESULTS_FOLDER"



# Run TAP vs prompt defense test (no defense)
python evaluation.py \
    --max-samples $MAX_SAMPLES \
    --dataset-path $DATASET_PATH \
    --results-folder "${RESULTS_FOLDER}_gpt4.1_${TARGET_MODEL}_" \
    --branching-factor 4 \
    --width 10 \
    --depth 10 \
    --attack-max-tokens 500 \
    --target-max-tokens 150 \
    --evaluator-model "gpt-4.1" \
    --attack-model "vicuna-13b" \
    --target-model $TARGET_MODEL \
    --no-wandb \

echo "defense test complete!"


MAX_SAMPLES=50  
DATASET_PATH="./data/advbench_subset.csv"
RESULTS_FOLDER="results/$(date +%Y%m%d_%H%M%S)"

TARGET_MODEL="vicuna-13b"

echo "Test parameters:"
echo "  Max samples: $MAX_SAMPLES"
echo "  Dataset: $DATASET_PATH"
echo "  Results folder: $RESULTS_FOLDER"



# Run TAP vs defense test (defense enabled)
python tap_prompt_attack_eval_back.py \
    --enable-defense \
    --defense-method "BackTranslation" \
    --max-samples $MAX_SAMPLES \
    --dataset-path $DATASET_PATH \
    --results-folder "${RESULTS_FOLDER}_gpt4.1_${TARGET_MODEL}_" \
    --branching-factor 4 \
    --width 10 \
    --depth 10 \
    --attack-max-tokens 500 \
    --target-max-tokens 150 \
    --evaluator-model "gpt-4.1" \
    --attack-model "vicuna-13b" \
    --target-model $TARGET_MODEL \
    --no-wandb \

echo "defense test complete!"


MAX_SAMPLES=50 
DATASET_PATH="./data/advbench_subset.csv"
RESULTS_FOLDER="results/$(date +%Y%m%d_%H%M%S)"

DEFENSE_MODEL_PATH="replace_with_your_embedding_model_path"

TARGET_MODEL="vicuna-13b"


# Run TAP vs defense test (defense enabled) with our embedding-based defense
python tap_prompt_attack_eval_back_database.py \
    --enable-defense \
    --enable-embedding-defense \
    --embedding-ckpt-path $DEFENSE_MODEL_PATH \
    --embedding-defense-threshold 0.5 \
    --embedding-device "cuda" \
    --defense-method "BackTranslation" \
    --max-samples $MAX_SAMPLES \
    --dataset-path $DATASET_PATH \
    --results-folder "${RESULTS_FOLDER}_gpt4.1_with_ours_database_${TARGET_MODEL}" \
    --branching-factor 4 \
    --width 10 \
    --depth 10 \
    --attack-max-tokens 500 \
    --target-max-tokens 150 \
    --evaluator-model "gpt-4.1" \
    --attack-model "vicuna-13b" \
    --target-model $TARGET_MODEL \
    --no-wandb \

echo "defense test complete!"