from dataclasses import dataclass, field
import os
from typing import List, Dict, Tuple, Optional

import sys
sys.path.append(os.environ.get("TAP_EVAL_PATH"))

from utils.common import conv_template
import config.ModelConfig as config

from Defense.base import BaseDefenderConfig, BaseDefender

import random


class RpoDefenderConfig(BaseDefenderConfig):
    """RPO Defense Configuration"""
    defender_cls: str = "RpoDefender"
    defender_name: str = "RpoDefender"

    # from  https://github.com/lapisrocks/rpo
    suffix: List[str] = [ ":}</'])language seems scary-----------}] estimation >>begin &=&pgf degli ({\ $|\textwidthAI NOTE:"]


class RpoDefender(BaseDefender):
    """RPO Defense Mechanism"""

    def __init__(self, config: RpoDefenderConfig):
        self.config = config
        self.suffix = config.suffix

        self.target_model = None

        # Defense statistics
        self.defense_stats = {
            "total_queries": 0,
            "defended_queries": 0,
            "suffix_used": len(self.suffix)
        }

        print(f"RPO defender initialized with {len(self.suffix)} optimized suffixes")

    def get_response(self, prompts_list):
        batchsize = len(prompts_list)
        convs_list = [conv_template(self.target_model.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            
            # Randomly select a suffix and append it to the prompt
            prompt = prompt + self.suffix[random.randint(0, len(self.suffix) - 1)]
            
            # User prompt
            conv.append_message(conv.roles[0], prompt)
            
            if "gpt" in self.target_model.model_name:
                # OpenAI does not have separators
                full_prompts.append(conv.to_openai_api_messages())
            elif "palm" in self.target_model.model_name:
                full_prompts.append(conv.messages[-1][1])
            else:
                conv.append_message(conv.roles[1], None) 
                full_prompts.append(conv.get_prompt())

        # print("======================")
        # print(full_prompts[0])
        # sys.exit()
        # Query the attack LLM in batched-queries with at most MAX_PARALLEL_STREAMS-many queries at a time
        outputs_list = []
        for left in range(0, len(full_prompts), config.MAX_PARALLEL_STREAMS):
            right = min(left + config.MAX_PARALLEL_STREAMS, len(full_prompts))

            if right == left:
                continue 

            print(f'\tQuerying target with {len(full_prompts[left:right])} prompts', flush=True)


            outputs_list.extend(
                                self.target_model.model.batched_generate(full_prompts[left:right], 
                                                            max_n_tokens = self.target_model.max_n_tokens,  
                                                            temperature = self.target_model.temperature,
                                                            top_p = self.target_model.top_p
                                                        )
            )
            
        return outputs_list
    

    def get_defense_stats(self) -> Dict:
        """Get defense statistics"""
        return self.defense_stats.copy()
    
    def reset_stats(self):
        """Reset defense statistics"""
        self.defense_stats["total_queries"] = 0
        self.defense_stats["defended_queries"] = 0

    
