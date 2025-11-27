"""
Simplified Metrics Tracking for Academic Project.

Removed Prometheus/Grafana dependencies. 
Uses simple file-based logging and tensorboard for metrics.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict
from loguru import logger
import numpy as np


class SimpleMetricsTracker:
    """
    Lightweight metrics tracker without external dependencies.
    
    Tracks training metrics and saves to JSON files for later analysis.
    Works seamlessly with TensorBoard for visualization.
    """
    
    def __init__(self, log_dir: str = "logs/metrics"):
        """
        Args:
            log_dir: Directory to save metrics JSON files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory metrics storage
        self.metrics: Dict[str, list] = defaultdict(list)
        self.start_time = time.time()
        self.episode_count = 0
        self.step_count = 0
        
        logger.info(f"SimpleMetricsTracker initialized at {self.log_dir}")
    
    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a scalar metric.
        
        Args:
            name: Metric name (e.g., 'train/reward')
            value: Metric value
            step: Step number (uses internal counter if None)
        """
        if step is None:
            step = self.step_count
        
        self.metrics[name].append({
            'step': step,
            'value': value,
            'timestamp': time.time()
        })
    
    def log_episode_metrics(self, metrics: Dict[str, Any]):
        """
        Log metrics for an entire episode.
        
        Args:
            metrics: Dictionary of episode metrics
        """
        self.episode_count += 1
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f'episode/{key}', value, self.episode_count)
    
    def log_step_metrics(self, metrics: Dict[str, Any]):
        """
        Log metrics for a single step.
        
        Args:
            metrics: Dictionary of step metrics
        """
        self.step_count += 1
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f'step/{key}', value, self.step_count)
    
    def get_metric_summary(self, name: str, window: int = 100) -> Dict[str, float]:
        """
        Get summary statistics for a metric.
        
        Args:
            name: Metric name
            window: Number of recent values to consider
            
        Returns:
            Dictionary with mean, std, min, max
        """
        if name not in self.metrics or not self.metrics[name]:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
        
        values = [m['value'] for m in self.metrics[name][-window:]]
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    
    def save_to_file(self, filename: Optional[str] = None):
        """
        Save all metrics to JSON file.
        
        Args:
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = self.log_dir / filename
        
        # Prepare data for JSON serialization
        data = {
            'start_time': self.start_time,
            'end_time': time.time(),
            'duration': time.time() - self.start_time,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'metrics': dict(self.metrics)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")
    
    def load_from_file(self, filename: str):
        """
        Load metrics from JSON file.
        
        Args:
            filename: Input filename
        """
        filepath = self.log_dir / filename
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.start_time = data['start_time']
        self.episode_count = data['episode_count']
        self.step_count = data['step_count']
        self.metrics = defaultdict(list, data['metrics'])
        
        logger.info(f"Metrics loaded from {filepath}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dictionary with all summary statistics
        """
        summary = {
            'duration': time.time() - self.start_time,
            'episodes': self.episode_count,
            'steps': self.step_count,
            'metrics': {}
        }
        
        # Add summary for each metric
        for metric_name in self.metrics.keys():
            summary['metrics'][metric_name] = self.get_metric_summary(metric_name)
        
        return summary
    
    def print_summary(self):
        """Print training summary to console."""
        summary = self.get_training_summary()
        
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Duration: {summary['duration']:.2f} seconds")
        logger.info(f"Episodes: {summary['episodes']}")
        logger.info(f"Steps: {summary['steps']}")
        logger.info("")
        
        for metric_name, stats in summary['metrics'].items():
            if stats['count'] > 0:
                logger.info(f"{metric_name}:")
                logger.info(f"  Mean: {stats['mean']:.4f}")
                logger.info(f"  Std:  {stats['std']:.4f}")
                logger.info(f"  Min:  {stats['min']:.4f}")
                logger.info(f"  Max:  {stats['max']:.4f}")
                logger.info("")
        
        logger.info("=" * 60)


class MetricsTracker(SimpleMetricsTracker):
    """Alias for backward compatibility."""
    pass


# Global metrics instance
_global_metrics: Optional[SimpleMetricsTracker] = None


def get_metrics_tracker(log_dir: str = "logs/metrics") -> SimpleMetricsTracker:
    """
    Get or create global metrics tracker instance.
    
    Args:
        log_dir: Directory for metrics logs
        
    Returns:
        SimpleMetricsTracker instance
    """
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = SimpleMetricsTracker(log_dir)
    return _global_metrics


def reset_metrics_tracker():
    """Reset global metrics tracker."""
    global _global_metrics
    _global_metrics = None


__all__ = [
    'SimpleMetricsTracker',
    'MetricsTracker',
    'get_metrics_tracker',
    'reset_metrics_tracker'
]
