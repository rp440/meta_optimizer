import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from typing import List, Tuple, Dict
from collections import deque

from ..models.ppo_agent import PPOAgent
from ..utils.config import PPOConfig
from ..utils.logger import Logger

class MetaOptimizer:
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent = PPOAgent().to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=config.learning_rate)
        self.baseline_losses = deque(maxlen=config.baseline_window)
        self.logger = Logger()
        self.window_stats = deque(maxlen=20)

    def compute_gae(self, rewards: List[float], values: List[torch.Tensor],
                   next_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = []
        returns = []
        running_return = next_value.item()
        running_advantage = 0

        for r, v in zip(reversed(rewards), reversed(values)):
            running_return = r + self.config.gamma * running_return
            running_tderror = r + self.config.gamma * next_value.item() - v.item()
            running_advantage = (running_tderror +
                               self.config.gamma * self.config.gae_lambda *
                               running_advantage)
            advantages.insert(0, running_advantage)
            returns.insert(0, running_return)
            next_value = v

        return (torch.tensor(advantages, device=self.device),
                torch.tensor(returns, device=self.device))

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            mean, std, value = self.agent(state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            return action, log_prob, value

    def compute_reward(self, loss: float, accuracy: float, time_taken: float, is_best: bool) -> float:
        if self.baseline_losses:
            loss_baseline = np.mean(list(self.baseline_losses))
            improvement = (loss_baseline - loss) / loss_baseline
        else:
            improvement = 0.0

        self.baseline_losses.append(loss)
        
        reward = (-loss * self.config.reward_scaling +
                 improvement * 2.0 +
                 accuracy * 0.5 -
                 np.log1p(time_taken) * 0.1 +
                 float(is_best))

        return -10.0 if loss > 5.0 or np.isnan(loss) else reward

    def update_policy(self, transitions: Dict[str, torch.Tensor]) -> Dict[str, float]:
        states = transitions['states']
        actions = transitions['actions']
        old_log_probs = transitions['log_probs']
        rewards = transitions['rewards']
        values = transitions['values']

        next_value = torch.zeros(1, device=self.device)
        values_list = [v.squeeze() for v in values.split(1)]
        advantages, returns = self.compute_gae(rewards.tolist(), values_list, next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = TensorDataset(states, actions, old_log_probs, advantages, returns)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        metrics = []

        for _ in range(self.config.ppo_epochs):
            for batch in loader:
                state_batch, action_batch, old_log_prob_batch, advantage_batch, return_batch = batch
                dist = torch.distributions.Normal(mean, std)
                log_probs = dist.log_prob(action_batch).sum(-1)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - old_log_prob_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1-self.config.clip_ratio, 1+self.config.clip_ratio) * advantage_batch
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(-1), return_batch)
                loss = (policy_loss +
                       self.config.value_coef * value_loss -
                       self.config.entropy_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                metrics.append({
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy': entropy.item()
                })

        return {k: np.mean([m[k] for m in metrics]) for k in metrics[0].keys()}

    def evaluate(self, model: nn.Module, val_loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        model.train()
        return correct / total if total > 0 else 0

    def train_episode(self, model: nn.Module, train_loader: DataLoader,
                     val_loader: DataLoader, episode: int) -> Dict[str, torch.Tensor]:
        buffers = {
            'states': [], 'actions': [], 'rewards': [],
            'log_probs': [], 'values': []
        }
        
        previous_loss = None
        best_loss = float('inf')
        episode_start = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            step_start = time.time()
            
            # Forward pass and compute gradients
            data, target = data.to(self.device), target.to(self.device)
            model.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()

            # Compute gradient statistics
            grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            mean_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
            max_grad_norm = max(grad_norms) if grad_norms else 0.0

            # Get validation accuracy periodically
            val_acc = self.evaluate(model, val_loader) if batch_idx % 50 == 0 else 0.0

            # Create state tensor for PPO
            current_loss = loss.item()
            loss_scale = current_loss / (previous_loss if previous_loss is not None else current_loss)
            progress = float(batch_idx) / len(train_loader)
            
            # Check if current loss is best
            is_best = current_loss < best_loss
            if is_best:
                best_loss = current_loss
            
            # Get windowed statistics
            if self.window_stats:
                param_norm, grad_norm, prev_loss = self.window_stats[-1]
                param_norm = param_norm / (sum(p.data.norm().item() for p in model.parameters()) + 1e-8)
                grad_norm = grad_norm / (mean_grad_norm + 1e-8)
                loss_trend = prev_loss / (current_loss + 1e-8)
            else:
                param_norm, grad_norm, loss_trend = 1.0, 1.0, 1.0

            state = torch.tensor([
                loss_scale,                    # Relative loss change
                mean_grad_norm / (max_grad_norm + 1e-8),  # Normalized mean gradient
                max_grad_norm / (100 + max_grad_norm),    # Bounded max gradient
                val_acc,                       # Validation accuracy
                progress,                      # Training progress
                param_norm,                    # Relative parameter norm
                grad_norm,                     # Relative gradient norm
                loss_trend,                    # Loss trend
                float(is_best)                 # Best loss indicator
            ], device=self.device, dtype=torch.float32).unsqueeze(0)

            # Get action from PPO agent
            action, log_prob, value = self.get_action(state)
            
            # Apply adaptive learning rate with bounds
            base_lr = torch.sigmoid(action).item() * 0.1  # Convert to (0, 0.1) range
            adaptive_factor = np.sqrt(1.0 / (mean_grad_norm + 1e-8))  # Adapt to gradient magnitude
            lr = base_lr * min(adaptive_factor, 10.0)    # Bound the adaptation
            
            # Apply gradient update with gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
            for param in model.parameters():
                if param.grad is not None:
                    param.data.add_(param.grad, alpha=-lr)

            # Track samples processed
            samples_processed = (batch_idx + 1) * data.size(0)
            total_samples = len(train_loader.dataset)

            step_time = time.time() - step_start
            reward = self.compute_reward(current_loss, val_acc, step_time, is_best)

            # Store transition
            buffers['states'].append(state)
            buffers['actions'].append(action)
            buffers['rewards'].append(reward)
            buffers['log_probs'].append(log_prob)
            buffers['values'].append(value)

            # Logging
            self.logger.log({
                'train_loss': current_loss,
                'rewards': reward,
                'learning_rates': lr
            })

            if batch_idx % 50 == 0:
                self.logger.log({'val_accuracy': val_acc})

            # Print progress
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)} '
                      f'({samples_processed}/{total_samples} samples), '
                      f'Loss: {current_loss:.4f}, LR: {lr:.6f}')

            previous_loss = current_loss
            self.window_stats.append((
                sum(p.data.norm().item() for p in model.parameters()),
                mean_grad_norm,
                current_loss
            ))

        # Episode completion
        episode_time = time.time() - episode_start
        final_val_acc = self.evaluate(model, val_loader)

        print(f"Episode Statistics:")
        print(f"Total samples processed: {total_samples:,}")
        print(f"Episode time: {episode_time:.2f} seconds")
        print(f"Samples per second: {total_samples/episode_time:.2f}")
        print(f"Final validation accuracy: {final_val_acc:.4f}")
        print(f"Best loss achieved: {best_loss:.4f}")

        self.logger.log({
            'episode_times': episode_time,
            'episode_best_loss': best_loss,
            'samples_processed': total_samples
        })

        return {
            'states': torch.cat(buffers['states']),
            'actions': torch.cat(buffers['actions']),
            'rewards': torch.tensor(buffers['rewards'], device=self.device),
            'log_probs': torch.cat(buffers['log_probs']),
            'values': torch.cat(buffers['values'])
        }
