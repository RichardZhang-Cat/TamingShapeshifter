from dataclasses import dataclass, field

class BaseBenchConfig:
    """Base configuration class"""
    pass

@dataclass
class AdvBenchConfig(BaseBenchConfig):
    """AdvBench"""
    # TAP参数
    branching_factor: int = 4
    width: int = 10
    depth: int = 10
    n_streams: int = 5
    keep_last_n: int = 3
    
    # 模型参数
    attack_model: str = "vicuna-13b"
    target_model: str = "vicuna-13b"
    evaluator_model: str = "gpt-4.1"
    
    attack_max_tokens: int = 500
    target_max_tokens: int = 150
    max_n_attack_attempts: int = 5
    
    evaluator_max_tokens: int = 10
    evaluator_temperature: float = 0.0
    
    # 数据集参数
    dataset_name: str = "advbench_sub50"
    dataset_path: str = "data/advbench_subset.csv"
    max_samples: int = 50
    
    # 输出参数
    results_folder: str = "results/advbench_evaluation"
    save_intermediate: bool = True

@dataclass
class JailbreakBenchConfig(BaseBenchConfig):
    """JailbreakBench"""
    # TAP参数
    branching_factor: int = 4
    width: int = 10
    depth: int = 10
    n_streams: int = 5
    keep_last_n: int = 3

    # 模型参数
    attack_model: str = "vicuna-13b"
    target_model: str = "vicuna-13b"
    evaluator_model: str = "gpt-4.1"
    
    # 模型参数
    attack_max_tokens: int = 500
    target_max_tokens: int = 150
    max_n_attack_attempts: int = 5

    evaluator_max_tokens: int = 10
    temperature: float = 1.0
    evaluator_temperature: float = 0.0
    
    # 数据集参数
    dataset_name: str = "jailbreakbench"
    dataset_path: str = "data/jailbreakbench_subset.csv"
    max_samples: int = 50
    
    # 输出参数
    results_folder: str = "results/jailbreakbench_evaluation"
    save_intermediate: bool = True