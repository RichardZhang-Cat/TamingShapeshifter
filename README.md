# TAP vs Prompt Defense Benchmark

This repository benchmarks multiple defense methods against self-evolving automated jailbreak attacks. It includes implementations of TAP and PAIR for attacks, and defenses such as our defense framework, ICD, BackTranslation, SmoothLLM, PerplexityFilter, and RPO.

## Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended) and recent NVIDIA drivers
- Access credentials for any external LLM APIs you plan to use

## Installation
1) Create and activate a virtual environment (optional but recommended).
2) Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Configuration
- Set the project root (used by code to locate configs/models):
```bash
export TAP_EVAL_PATH="$(pwd)"
```
- Configure model paths and API endpoints in [config/ModelConfig.py](config/ModelConfig.py). Replace placeholder paths/links with your local or hosted models.
- For API-based models, set required keys/bases as environment variables before running (e.g., `OPENAI_API_KEY`, `OPENAI_API_BASE`, `CLAUDE_API_KEY`, `GEMINI_API_KEY`, etc.). Leave blank if not using a given provider. Example for OpenAI:
```bash
export OPENAI_API_KEY="your_openai_key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # or your proxy base
```
- Set CUDA device selection:
```bash
export CUDA_VISIBLE_DEVICES=0
```

## Running Evaluations
Example: run TAP vs prompt defenses without defense:
```bash
python evaluation.py \
	--max-samples 50 \
	--dataset-path ./data/advbench_subset.csv \
	--results-folder results/run_$(date +%Y%m%d_%H%M%S) \
	--branching-factor 4 \
	--width 10 \
	--depth 10 \
	--attack-max-tokens 500 \
	--target-max-tokens 150 \
	--evaluator-model gpt-4.1 \
	--attack-model vicuna-13b \
	--target-model vicuna-13b
```

Enable a defense (example: BackTranslation):
```bash
python tap_prompt_attack_eval_back.py \
	--enable-defense \
	--defense-method BackTranslation \
	--max-samples 50 \
	--dataset-path ./data/advbench_subset.csv \
	--results-folder results/run_bt_$(date +%Y%m%d_%H%M%S) \
	--branching-factor 4 \
	--width 10 \
	--depth 10 \
	--attack-max-tokens 500 \
	--target-max-tokens 150 \
	--evaluator-model gpt-4.1 \
	--attack-model vicuna-13b \
	--target-model vicuna-13b
```

Enable embedding-based defense (provide your checkpoint):
```bash
python tap_prompt_attack_eval_back_database.py \
	--enable-defense \
	--enable-embedding-defense \
	--embedding-ckpt-path /path/to/your/embedding/model.ckpt \
	--embedding-defense-threshold 0.5 \
	--embedding-device cuda \
	--defense-method BackTranslation \
	--max-samples 50 \
	--dataset-path ./data/advbench_subset.csv \
	--results-folder results/run_bt_emb_$(date +%Y%m%d_%H%M%S) \
	--branching-factor 4 \
	--width 10 \
	--depth 10 \
	--attack-max-tokens 500 \
	--target-max-tokens 150 \
	--evaluator-model gpt-4.1 \
	--attack-model vicuna-13b \
	--target-model vicuna-13b
```

You can also run the convenience script [test.sh](test.sh) after adjusting env vars and paths inside it.

## Tips
- Reduce `--max-samples` and widths/depths for quick smoke tests.
- Ensure all model paths in `ModelConfig.py` exist or use API endpoints instead.
- Monitor GPU memory; lower `--width`, `--depth`, or `--branching-factor` if you hit OOM.
- Set `TAP_EVAL_PATH` consistently when launching from different directories.

## Outputs
- Results are written to the folder specified by `--results-folder`, including intermediate artifacts if not disabled.
- If WandB logging is enabled (default), runs will be logged to your configured WandB project; set `--no-wandb` to disable.



