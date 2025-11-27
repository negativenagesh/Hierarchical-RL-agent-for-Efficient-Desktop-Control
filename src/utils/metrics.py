"""Metrics tracking utilities"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
from typing import Dict, Any, Optional
from collections import deque
import numpy as np


class MetricsTracker:
    """
    Metrics tracking using Prometheus
    
    Tracks training and inference metrics for monitoring.
    """
    
    def __init__(self):
        # Inference metrics
        self.inference_counter = Counter(
            'agent_inference_total',
            'Total number of inference requests'
        )
        
        self.inference_duration = Histogram(
            'agent_inference_duration_seconds',
            'Inference duration in seconds',
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        # Execution metrics
        self.execution_counter = Counter(
            'agent_execution_total',
            'Total number of action executions',
            ['action_type']
        )
        
        self.execution_success_counter = Counter(
            'agent_execution_success_total',
            'Total successful executions',
            ['action_type']
        )
        
        # Training metrics
        self.training_step = Gauge(
            'training_step',
            'Current training step'
        )
        
        self.training_reward = Gauge(
            'training_episode_reward',
            'Episode reward'
        )
        
        self.training_success_rate = Gauge(
            'training_success_rate',
            'Success rate'
        )
        
        self.policy_loss = Gauge(
            'training_policy_loss',
            'Policy loss'
        )
        
        self.value_loss = Gauge(
            'training_value_loss',
            'Value loss'
        )
        
        # Local tracking
        self.inference_times: deque = deque(maxlen=1000)
        self.episode_rewards: deque = deque(maxlen=100)
        self.episode_successes: deque = deque(maxlen=100)
        
    def record_inference(self, duration: float) -> None:
        """Record inference request"""
        self.inference_counter.inc()
        self.inference_duration.observe(duration)
        self.inference_times.append(duration)
    
    def record_execution(self, action_type: str, success: bool) -> None:
        """Record action execution"""
        self.execution_counter.labels(action_type=action_type).inc()
        if success:
            self.execution_success_counter.labels(action_type=action_type).inc()
    
    def record_training_step(
        self,
        step: int,
        episode_reward: Optional[float] = None,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None
    ) -> None:
        """Record training step"""
        self.training_step.set(step)
        
        if episode_reward is not None:
            self.training_reward.set(episode_reward)
            self.episode_rewards.append(episode_reward)
        
        if policy_loss is not None:
            self.policy_loss.set(policy_loss)
        
        if value_loss is not None:
            self.value_loss.set(value_loss)
    
    def record_episode(self, reward: float, success: bool) -> None:
        """Record episode completion"""
        self.episode_rewards.append(reward)
        self.episode_successes.append(float(success))
        
        # Update success rate
        if len(self.episode_successes) > 0:
            success_rate = np.mean(list(self.episode_successes))
            self.training_success_rate.set(success_rate)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            'inference': {
                'total': int(self.inference_counter._value.get()),
                'avg_duration_ms': np.mean(list(self.inference_times)) * 1000 if self.inference_times else 0
            },
            'training': {
                'step': int(self.training_step._value.get()),
                'avg_reward': np.mean(list(self.episode_rewards)) if self.episode_rewards else 0,
                'success_rate': np.mean(list(self.episode_successes)) if self.episode_successes else 0
            }
        }
    
    def get_prometheus_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        return generate_latest()


# Global metrics instance
metrics_tracker = MetricsTracker()
