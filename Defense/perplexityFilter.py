from dataclasses import dataclass, field
import traceback
from typing import List, Dict, Tuple, Optional
import math

import sys

import torch

import os
sys.path.append(os.environ.get("TAP_EVAL_PATH"))

from utils.log_likelihood import get_log_likelihood, get_log_prob
from utils.common import conv_template
import config.ModelConfig as config

from Defense.base import BaseDefenderConfig, BaseDefender


@dataclass
class PerplexityFilterConfig(BaseDefenderConfig):
    """Perplexity Filter Defense Configuration"""
    defender_cls: str = "PerplexityFilterDefender"
    defender_name: str = "PerplexityFilter"
    
    # Perplexity threshold: queries exceeding this value will be rejected
    perplexity_threshold: float = 43.0223
     
    # Whether to use the target model itself as the perplexity evaluator
    use_target_as_filter: bool = True
    
    # Rejection response template
    reject_response: str = "sorry, I can't assist with that."


class PerplexityFilterDefender(BaseDefender):
    """
    Perplexity Filter Defense Mechanism
    
    Core Idea:
    Adversarial attacks or jailbreak prompts are usually unnatural in language, resulting in high perplexity.
    By calculating the perplexity of the input and comparing it with a threshold, potentially malicious inputs are filtered out.
    
    Perplexity = exp(-mean(log_probabilities))
    - Lower perplexity → More natural text
    - Higher perplexity → More unnatural text (possibly an attack)
    """

    def __init__(self, config: PerplexityFilterConfig):
        self.config = config
        self.threshold = config.perplexity_threshold
        self.reject_response = config.reject_response
        self.use_target_as_filter = config.use_target_as_filter

        self.target_model = None  # Will be set externally
        self.filter_llm = None    # Model used for calculating perplexity

        # Defense statistics
        self.defense_stats = {
            "total_queries": 0,
            "defended_queries": 0,  # Number of rejected queries
            "passed_queries": 0,    # Number of passed queries
            "avg_perplexity": 0.0,
            "max_perplexity": 0.0,
            "min_perplexity": float('inf'),
            "perplexities": []  # Record all perplexity values
        }

        print(f"Perplexity Filter Defender initialized")
        print(f"  Perplexity threshold: {self.threshold}")
        print(f"  Use target model as filter: {self.use_target_as_filter}")


    def _calculate_perplexity(self, log_probs: torch.Tensor) -> float:
        """
        Calculate perplexity
        
        Args:
            log_probs: tensor of shape (seq_len, ) containing log probabilities

        Returns:
            perplexity: Perplexity value
            
        Formula: Perplexity = exp(-mean(log_probs))
        """
        if log_probs is None:
            print("Warning: No log probabilities available, returning infinite perplexity")
            return float('inf')
        
        # Calculate mean log probability (here CE is calculated with negative sign)
        mean_log_prob = torch.mean(log_probs)
        
        # Calculate perplexity
        perplexity = math.exp(mean_log_prob)
        
        return perplexity

    def _evaluate_perplexity(self, prompt: str) -> Tuple[float, List[float]]:
        """
        Evaluate the perplexity of a single prompt
        
        Args:
            prompt: User input prompt
            
        Returns:
            perplexity: Perplexity value
            log_probs: List of log probabilities
        """
        # Use filter model (or target model) to calculate log probabilities
        filter_model = self.filter_llm if self.filter_llm else self.target_model
        
        if filter_model is None:
            print("Warning: No filter model available, passing by default")
            return 0.0, []
        
        try:
            # Create conversation template
            conv = conv_template(filter_model.template)
            conv.append_message(conv.roles[0], prompt)

            
            # Call model to get log probabilities
            
            # log_probs = get_log_likelihood(conv, filter_model.model)
            log_probs = get_log_prob(conv, filter_model.model)
            
            # Calculate perplexity
            perplexity = self._calculate_perplexity(log_probs)
            
            return perplexity, log_probs
            
        except Exception as e:
            print(f"Error calculating perplexity: {e}")
            
            traceback.print_exc()
            return 0.0, []  # Pass by default on error

    def _should_reject(self, prompt: str) -> Tuple[bool, float, str]:
        """
        Determine whether the query should be rejected
        
        Args:
            prompt: User input
            
        Returns:
            should_reject: Whether to reject
            perplexity: Perplexity value
            reason: Reason for rejection
        """
        # Calculate perplexity
        perplexity, log_probs = self._evaluate_perplexity(prompt)
        
        # Update statistics
        self.defense_stats["perplexities"].append(perplexity)
        self.defense_stats["max_perplexity"] = max(
            self.defense_stats["max_perplexity"], 
            perplexity
        )
        self.defense_stats["min_perplexity"] = min(
            self.defense_stats["min_perplexity"], 
            perplexity
        )
        
        # Determine if threshold is exceeded
        if perplexity > self.threshold:
            reason = f"Perplexity {perplexity:.2f} exceeds threshold {self.threshold:.2f}, suspected malicious input"
            return True, perplexity, reason
        else:
            reason = f"Perplexity {perplexity:.2f} below threshold, normal input"
            return False, perplexity, reason

    def get_response(self, prompts_list: List[str]) -> List[str]:
        """
        Apply perplexity filtering to input prompt list and return responses
        
        Args:
            prompts_list: List of user input prompts
            
        Returns:
            outputs_list: List of responses (rejected queries return rejection message, passed queries return model response)
        """
        batchsize = len(prompts_list)
        self.defense_stats["total_queries"] += batchsize
        
        # Store prompts that pass the filter
        passed_prompts = []
        passed_indices = []
        perplexity_results = []
        
        print(f"\n🛡️ Perplexity Filter Defense started (total {batchsize} queries)")
        
        # Calculate perplexity for each prompt and determine
        for i, prompt in enumerate(prompts_list):
            should_reject, perplexity, reason = self._should_reject(prompt)
            
            perplexity_results.append({
                "index": i,
                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                "perplexity": perplexity,
                "should_reject": should_reject,
                "reason": reason
            })
            
            if should_reject:
                self.defense_stats["defended_queries"] += 1
                print(f"  ❌ Query {i}: perplexity={perplexity:.2f} > {self.threshold:.2f} → Rejected")
            else:
                self.defense_stats["passed_queries"] += 1
                passed_prompts.append(prompt)
                passed_indices.append(i)
                print(f"  ✅ Query {i}: perplexity={perplexity:.2f} ≤ {self.threshold:.2f} → Passed")
        
        # Update average perplexity
        if self.defense_stats["perplexities"]:
            self.defense_stats["avg_perplexity"] = sum(
                self.defense_stats["perplexities"]
            ) / len(self.defense_stats["perplexities"])
        
        print(f"📊 Filtering result: {len(passed_prompts)}/{batchsize} queries passed")
        
        # If no queries passed, return rejection message for all
        if not passed_prompts:
            print("⚠️ All queries rejected")
            return [self.reject_response] * batchsize
        
        # Generate responses for passed queries
        print(f"🤖 Generating responses for {len(passed_prompts)} passed queries...")

        passed_outputs = self.target_model.get_response(passed_prompts)
        
        
        # Reconstruct responses: fill rejected positions with rejection message, passed positions with model response
        outputs_list = []
        passed_output_idx = 0
        
        for i in range(batchsize):
            if i in passed_indices:
                outputs_list.append(passed_outputs[passed_output_idx])
                passed_output_idx += 1
            else:
                outputs_list.append(self.reject_response)
        
        return outputs_list

    def get_defense_stats(self) -> Dict:
        """Get defense statistics"""
        stats = self.defense_stats.copy()
        
        # Calculate defense rate
        if stats["total_queries"] > 0:
            stats["defense_rate"] = stats["defended_queries"] / stats["total_queries"]
            stats["pass_rate"] = stats["passed_queries"] / stats["total_queries"]
        else:
            stats["defense_rate"] = 0.0
            stats["pass_rate"] = 0.0
        
        # Remove detailed perplexity list (too long)
        stats["perplexity_count"] = len(stats["perplexities"])
        del stats["perplexities"]
        
        return stats
    
    def reset_stats(self):
        """Reset defense statistics"""
        self.defense_stats = {
            "total_queries": 0,
            "defended_queries": 0,
            "passed_queries": 0,
            "avg_perplexity": 0.0,
            "max_perplexity": 0.0,
            "min_perplexity": float('inf'),
            "perplexities": []
        }
        print("Defense statistics reset")
