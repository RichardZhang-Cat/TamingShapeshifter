#!/usr/bin/env python3

import os
import sys
import json
import time
import csv
import copy
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import torch
import random

import numpy as np




os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
tap_eval_path = os.environ.get("TAP_EVAL_PATH")
sys.path.append(tap_eval_path)

from Defense.perplexityFilter import PerplexityFilterConfig

from config.BenchConfig import AdvBenchConfig
from Defense.icl import IclDefender, IclDefenderConfig
from Defense.RPO import RpoDefender, RpoDefenderConfig
from Defense.perplexityFilter import PerplexityFilterConfig

from Defense.backTranslation import BackTranslationConfig

from Defense.smoothLLM import SmoothLLMDefenderConfig

from Defense.ours import EmbeddingDefenseConfig

from Attack.TAP import TapEvaluator


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate TAP attack vs prompt-based defense methods")
    
    # TAP parameters
    parser.add_argument("--branching-factor", type=int, default=4, help="Branching factor")
    parser.add_argument("--width", type=int, default=10, help="Search width")
    parser.add_argument("--depth", type=int, default=10, help="Search depth")
    parser.add_argument("--n-streams", type=int, default=5, help="Number of initial streams")
    
    # Model parameters
    parser.add_argument("--attack-model", type=str, default="vicuna-13b", help="Attack LLM model")
    parser.add_argument("--attack-max-tokens", type=int, default=500, help="Max tokens for attack model")
    parser.add_argument("--max-n-attack-attempts", type=int, default=5, help="Max attack attempts")
    
    parser.add_argument("--target-model", type=str, default="vicuna-13b", help="Target LLM model")
    parser.add_argument("--target-max-tokens", type=int, default=150, help="Max tokens for target model")
    # parser.add_argument("--temperature", type=float, default=1.0, help="Generation temperature")

    parser.add_argument("--evaluator-model", type=str, default="gpt-4.1", help="Evaluator model")
    parser.add_argument("--eval-temperature", type=float, default=0.0, help="Evaluator temperature")
    
    # Dataset parameters
    parser.add_argument("--dataset-path", type=str, default="../data/advbench_subset.csv", help="Dataset path")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum number of samples")
    
    # Defense parameters
    parser.add_argument("--enable-defense", action="store_true", help="Enable defense mechanisms")
    parser.add_argument("--defense-method", type=str, default="icl", choices=["icl", "self_reminder", "rpo", "PerplexityFilter", "BackTranslation", "SmoothLLM", "ICL_BackTranslation" ], help="Defense method")

    # Embedding defense
    parser.add_argument("--enable-embedding-defense", action="store_true", help="Enable embedding-based defense")
    parser.add_argument("--embedding-ckpt-path", type=str, default="", help="Embedding defense checkpoint path")
    parser.add_argument("--embedding-defense-threshold", type=float, default=0.4, help="Embedding defense threshold")
    parser.add_argument("--embedding-device", type=str, default="cuda", help="Embedding defense device")
    
    # Output parameters
    parser.add_argument("--results-folder", type=str, default="results_tap_prompt_defense_evaluation", help="Results folder")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WandB logging")
    parser.add_argument("--no-save-intermediate", action="store_true", help="Disable saving intermediate results")
    
    args = parser.parse_args()

    print("Launch parameters:")
    print(f"  Defense enabled: {args.enable_defense}")

    print(f"  Dataset path: {args.dataset_path}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Results folder: {args.results_folder}")
    
    # Create configuration
    config = AdvBenchConfig(
        branching_factor=args.branching_factor,
        width=args.width,
        depth=args.depth,
        n_streams=args.n_streams,
        attack_model=args.attack_model,
        target_model=args.target_model,
        evaluator_model=args.evaluator_model,
        attack_max_tokens=args.attack_max_tokens,
        target_max_tokens=args.target_max_tokens,
        max_n_attack_attempts=args.max_n_attack_attempts,
        evaluator_temperature=args.eval_temperature,
        dataset_path=args.dataset_path,
        max_samples=args.max_samples,
        results_folder=args.results_folder,
        save_intermediate=not args.no_save_intermediate
    )

    if args.defense_method == "icl":
        # Create ICL defense configuration
        defense_config = IclDefenderConfig()
    elif args.defense_method == "rpo":
        defense_config = RpoDefenderConfig()
    elif args.defense_method == "PerplexityFilter":
        defense_config = PerplexityFilterConfig()
    elif args.defense_method == "BackTranslation":
        defense_config = BackTranslationConfig()
    elif args.defense_method == "SmoothLLM":
        defense_config = SmoothLLMDefenderConfig()

    else:
        raise ValueError(f"Unsupported defense method: {args.defense_method}")

    embedding_defense_config = None
    if args.enable_embedding_defense:
        embedding_defense_config = EmbeddingDefenseConfig(
            model_path=args.embedding_ckpt_path,
            similarity_threshold=args.embedding_defense_threshold,
            device=args.embedding_device
        )

    # Create evaluator and run
    evaluator = TapEvaluator(
        config, 
        use_wandb=not args.no_wandb,
        enable_defense=args.enable_defense,
        defense_config=defense_config,
        enable_embedding_defense=args.enable_embedding_defense,
        embedding_defense_config=embedding_defense_config
    )
    
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()