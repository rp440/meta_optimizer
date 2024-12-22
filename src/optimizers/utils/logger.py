
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

class Logger:
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_accuracy': [],
            'rewards': [],
            'episode_times': [],
            'episode_best_loss': [],
            'learning_rates': [],
            'samples_processed': []
        }
        self.start_time = datetime.now()

    def log(self, metrics_dict: dict):
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def plot_metrics(self, save_path='training_metrics.png'):
        plt.rcParams.update({'font.size': 10})
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics Overview', fontsize=14, y=0.95)

        self._plot_loss(axes[0, 0])
        self._plot_accuracy(axes[0, 1])
        self._plot_rewards(axes[1, 0])
        self._plot_times(axes[1, 1])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self._print_summary()

    def _plot_loss(self, ax):
        if self.metrics['episode_best_loss']:
            ax.plot(self.metrics['episode_best_loss'], 'b-', linewidth=2)
            ax.set_title('Best Loss per Episode')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.grid(True, linestyle='--', alpha=0.7)

    def _plot_accuracy(self, ax):
        if self.metrics['val_accuracy']:
            ax.plot(self.metrics['val_accuracy'], 'g-', linewidth=2)
            ax.set_title('Validation Accuracy')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Accuracy')
            ax.grid(True, linestyle='--', alpha=0.7)

    def _plot_rewards(self, ax):
        if self.metrics['rewards']:
            ax.plot(self.metrics['rewards'], 'r-', linewidth=2)
            ax.set_title('Rewards per Episode')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.grid(True, linestyle='--', alpha=0.7)

    def _plot_times(self, ax):
        if self.metrics['episode_times']:
            ax.plot(self.metrics['episode_times'], color='orange', linewidth=2)
            ax.set_title('Episode Duration')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Time (seconds)')
            ax.grid(True, linestyle='--', alpha=0.7)

    def _print_summary(self):
        total_time = (datetime.now() - self.start_time).total_seconds() / 60
        print("Training Summary:")
        print(f"Total training time: {total_time:.2f} minutes")
        if self.metrics['val_accuracy']:
            print(f"Best validation accuracy: {max(self.metrics['val_accuracy']):.4f}")
        if self.metrics['episode_best_loss']:
            print(f"Best loss achieved: {min(self.metrics['episode_best_loss']):.4f}")
        if self.metrics['episode_times']:
            print(f"Average episode time: {np.mean(self.metrics['episode_times']):.2f} seconds")
        if self.metrics['samples_processed']:
            total_samples = sum(self.metrics['samples_processed'])
            print(f"Total samples processed: {total_samples:,}")
            print(f"Average samples/second: {total_samples/(total_time*60):.2f}")
