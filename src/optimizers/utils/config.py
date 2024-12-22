
from dataclasses import dataclass

@dataclass
class PPOConfig:
    """Configuration for PPO algorithm"""
    steps_per_episode: int = 250
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    ppo_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    target_kl: float = 0.01
    max_grad_norm: float = 0.5
    reward_scaling: float = 2.0
    baseline_window: int = 10
