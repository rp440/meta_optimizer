
import os
import torch
import numpy as np
import time
from optimizers.models.cnn import CIFARCNN
from optimizers.optimizers.meta_optimizer import MetaOptimizer
from optimizers.utils.config import PPOConfig
from optimizers.utils.data import get_cifar10_loaders

def train_meta_optimizer():
    """Main training function"""
    # Training parameters
    num_episodes = 5
    save_interval = 1
    batch_size = 128

    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)

    # Initialize components
    config = PPOConfig()
    meta_optimizer = MetaOptimizer(config)
    train_loader, val_loader = get_cifar10_loaders(batch_size)

    best_val_acc = 0.0
    best_model_path = None

    print(f"Starting training on device: {meta_optimizer.device}")

    try:
        for episode in range(num_episodes):
            print(f"\nStarting Episode {episode + 1}/{num_episodes}")
            episode_start = time.time()

            # Initialize new CNN model for this episode
            model = CIFARCNN().to(meta_optimizer.device)

            # Train for one episode
            transitions = meta_optimizer.train_episode(
                model,
                train_loader,
                val_loader,
                episode
            )

            # Update meta-optimizer policy
            update_metrics = meta_optimizer.update_policy(transitions)

            # Calculate episode statistics
            episode_time = time.time() - episode_start
            current_val_acc = meta_optimizer.logger.metrics['val_accuracy'][-1] if meta_optimizer.logger.metrics['val_accuracy'] else 0.0

            # Print episode summary
            print(f"\nEpisode {episode + 1} Summary:")
            print(f"Time: {episode_time:.2f}s")
            print(f"Best Loss: {meta_optimizer.logger.metrics['episode_best_loss'][-1]:.4f}")
            print(f"Validation Accuracy: {current_val_acc:.4f}")
            print(f"Average Reward: {np.mean(meta_optimizer.logger.metrics['rewards'][-50:]):.4f}")

            # Save best model
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                best_model_path = f"checkpoints/best_meta_optimizer.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'meta_optimizer_state_dict': meta_optimizer.agent.state_dict(),
                    'optimizer_state_dict': meta_optimizer.optimizer.state_dict(),
                    'episode': episode,
                    'val_acc': best_val_acc
                }, best_model_path)
                print(f"New best model saved! Validation Accuracy: {best_val_acc:.4f}")

            # Regular checkpoint saving
            if (episode + 1) % save_interval == 0:
                save_path = f"checkpoints/meta_optimizer_episode_{episode+1}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'meta_optimizer_state_dict': meta_optimizer.agent.state_dict(),
                    'optimizer_state_dict': meta_optimizer.optimizer.state_dict(),
                    'episode': episode
                }, save_path)
                print(f"Saved checkpoint to {save_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        # Plot final metrics
        meta_optimizer.logger.plot_metrics()
        print("\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        if best_model_path:
            print(f"Best model saved at: {best_model_path}")

if __name__ == "__main__":
    train_meta_optimizer()
