# Config file for local Vicuna-13B model
# Update the paths below to match your environment

# Local model paths - replace with your actual locations
VICUNA_7B_PATH = "path"  # Replace with your model path
VICUNA_13B_PATH = "path"  # If using the 13B variant
QWEN3_32B_PATH = "path"  # Qwen3-32B model path; update as needed
LLAMA_PATH = "path"  # If using a Llama model

MPNET_PATH = "path"  # If using an MPNet model


# API endpoints (for API deployments)
VICUNA_API_LINK = "https://..."  # If using an API-hosted version
LLAMA_API_LINK = "https://..."  # If using an API-hosted version

# Generation parameters
ATTACK_TEMP = 1.0      # Attack model temperature; higher is more random
TARGET_TEMP = 0.0      # Target model temperature; 0 means deterministic output
ATTACK_TOP_P = 0.9     # Attack model top-p
TARGET_TOP_P = 1.0     # Target model top-p

# Parallel streams - adjust based on GPU memory
MAX_PARALLEL_STREAMS = 10  # Reduce further if you hit OOM
