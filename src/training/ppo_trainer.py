"""PPO Trainer Implementation"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import time
from torch.utils.tensorboard import SummaryWriter

from ..agent.policy import HierarchicalPolicy
from ..environment.base_env import OSEnvironment
from .replay_buffer import ReplayBuffer


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) Trainer
    
    Implements PPO with clipped objective for training the hierarchical agent.
    """
    
    def __init__(
        self,
        policy: HierarchicalPolicy,
        env: OSEnvironment,
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        buffer_size: int = 2048,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        log_dir: str = 'logs'
    ):
        """
        Args:
            policy: Hierarchical policy to train
            env: Training environment
            learning_rate: Learning rate for optimizer
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            max_grad_norm: Maximum gradient norm for clipping
            update_epochs: Number of epochs per update
            num_minibatches: Number of minibatches per epoch
            buffer_size: Replay buffer size
            device: Device to train on
            log_dir: Directory for tensorboard logs
        """
        self.policy = policy.to(device)
        self.env = env
        self.device = torch.device(device)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.buffer_size = buffer_size
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        self.episode_count = 0
        
        # Statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.success_rate: List[bool] = []
        
    def collect_rollouts(self, num_steps: int) -> Dict[str, Any]:
        """
        Collect rollouts from the environment
        
        Args:
            num_steps: Number of steps to collect
        Returns:
            Rollout statistics
        """
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_success = False
        
        for step in range(num_steps):
            # Convert observation to tensor
            state_dict = self._prepare_state(obs)
            
            # Sample action from policy
            with torch.no_grad():
                action_type, coordinates, log_prob, value = self.policy.sample_action(state_dict)
            
            # Convert action to dictionary for environment
            action = {
                'action_type': action_type.cpu().numpy()[0],
                'coordinates': coordinates.cpu().numpy()[0]
            }
            
            # Execute action
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.buffer.add(
                state=obs,
                action_type=action_type.cpu().numpy()[0],
                coordinates=coordinates.cpu().numpy()[0],
                reward=reward,
                value=value.cpu().numpy()[0][0],
                log_prob=log_prob.cpu().numpy()[0],
                done=done
            )
            
            episode_reward += reward
            episode_length += 1
            self.global_step += 1
            
            # Update for next step
            obs = next_obs
            
            # Handle episode end
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episode_success = info.get('success', False)
                self.success_rate.append(episode_success)
                self.episode_count += 1
                
                # Log episode statistics
                self.writer.add_scalar('rollout/episode_reward', episode_reward, self.global_step)
                self.writer.add_scalar('rollout/episode_length', episode_length, self.global_step)
                self.writer.add_scalar('rollout/success', float(episode_success), self.global_step)
                
                # Reset environment
                obs, info = self.env.reset()
                episode_reward = 0
                episode_length = 0
        
        # Compute advantages and returns
        self.buffer.compute_advantages(self.gamma, self.gae_lambda)
        
        # Return statistics
        stats = {
            'mean_reward': np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0,
            'success_rate': np.mean(self.success_rate[-100:]) if self.success_rate else 0
        }
        
        return stats
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using PPO
        
        Returns:
            Training statistics
        """
        # Get batch from buffer
        batch = self.buffer.get()
        
        batch_size = len(batch['rewards'])
        minibatch_size = batch_size // self.num_minibatches
        
        # Statistics
        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clip_fractions = []
        
        # Multiple epochs
        for epoch in range(self.update_epochs):
            # Shuffle indices
            indices = np.random.permutation(batch_size)
            
            # Minibatch updates
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                
                # Get minibatch
                mb_states = [batch['states'][i] for i in mb_indices]
                mb_action_types = torch.tensor(
                    [batch['action_types'][i] for i in mb_indices],
                    dtype=torch.long,
                    device=self.device
                )
                mb_coordinates = torch.tensor(
                    np.array([batch['coordinates'][i] for i in mb_indices]),
                    dtype=torch.float32,
                    device=self.device
                )
                mb_old_log_probs = torch.tensor(
                    [batch['log_probs'][i] for i in mb_indices],
                    dtype=torch.float32,
                    device=self.device
                )
                mb_advantages = torch.tensor(
                    [batch['advantages'][i] for i in mb_indices],
                    dtype=torch.float32,
                    device=self.device
                )
                mb_returns = torch.tensor(
                    [batch['returns'][i] for i in mb_indices],
                    dtype=torch.float32,
                    device=self.device
                )
                
                # Prepare states
                state_dicts = [self._prepare_state(s) for s in mb_states]
                
                # Evaluate actions
                log_probs, entropies_batch, values = [], [], []
                for state_dict, action_type, coords in zip(state_dicts, mb_action_types, mb_coordinates):
                    log_prob, entropy, value = self.policy.evaluate_actions(
                        state_dict,
                        action_type.unsqueeze(0),
                        coords.unsqueeze(0)
                    )
                    log_probs.append(log_prob)
                    entropies_batch.append(entropy)
                    values.append(value)
                
                log_probs = torch.cat(log_probs)
                entropies_batch = torch.cat(entropies_batch)
                values = torch.cat(values).squeeze()
                
                # Compute ratio
                ratio = torch.exp(log_probs - mb_old_log_probs)
                
                # Compute surrogate losses
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * ((values - mb_returns) ** 2).mean()
                
                # Entropy bonus
                entropy_loss = -entropies_batch.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(-entropy_loss.item())
                
                # Approximate KL divergence
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    approx_kls.append(approx_kl.item())
                    
                    # Clip fraction
                    clip_fraction = ((ratio - 1).abs() > self.clip_epsilon).float().mean()
                    clip_fractions.append(clip_fraction.item())
        
        # Clear buffer
        self.buffer.clear()
        
        # Log training statistics
        stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'approx_kl': np.mean(approx_kls),
            'clip_fraction': np.mean(clip_fractions)
        }
        
        for key, value in stats.items():
            self.writer.add_scalar(f'train/{key}', value, self.global_step)
        
        return stats
    
    def train(
        self,
        total_timesteps: int,
        rollout_steps: int = 2048,
        save_interval: int = 10000,
        eval_interval: int = 5000,
        save_dir: str = 'checkpoints'
    ) -> None:
        """
        Main training loop
        
        Args:
            total_timesteps: Total training timesteps
            rollout_steps: Steps per rollout
            save_interval: Save checkpoint every N steps
            eval_interval: Evaluate every N steps
            save_dir: Directory to save checkpoints
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        num_updates = total_timesteps // rollout_steps
        
        print(f"Starting training for {total_timesteps} timesteps ({num_updates} updates)")
        print(f"Device: {self.device}")
        
        for update in range(num_updates):
            # Collect rollouts
            rollout_stats = self.collect_rollouts(rollout_steps)
            
            # Update policy
            train_stats = self.update_policy()
            
            # Logging
            if update % 10 == 0:
                elapsed_time = time.time() - start_time
                fps = self.global_step / elapsed_time
                
                print(f"\nUpdate {update}/{num_updates} | Step {self.global_step}/{total_timesteps}")
                print(f"FPS: {fps:.2f} | Time: {elapsed_time:.2f}s")
                print(f"Mean Reward: {rollout_stats['mean_reward']:.2f}")
                print(f"Success Rate: {rollout_stats['success_rate']:.2%}")
                print(f"Policy Loss: {train_stats['policy_loss']:.4f}")
                print(f"Value Loss: {train_stats['value_loss']:.4f}")
            
            # Save checkpoint
            if self.global_step % save_interval == 0:
                checkpoint_path = save_path / f"checkpoint_{self.global_step}.pt"
                self.save_checkpoint(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Evaluation
            if self.global_step % eval_interval == 0:
                eval_stats = self.evaluate(num_episodes=10)
                print(f"\nEvaluation: Success Rate = {eval_stats['success_rate']:.2%}")
                self.writer.add_scalar('eval/success_rate', eval_stats['success_rate'], self.global_step)
        
        # Final save
        final_path = save_path / "final_model.pt"
        self.save_checkpoint(final_path)
        print(f"\nTraining complete! Final model saved to {final_path}")
        
        self.writer.close()
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate policy"""
        self.policy.eval()
        
        episode_rewards = []
        episode_successes = []
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                state_dict = self._prepare_state(obs)
                
                with torch.no_grad():
                    action = self.policy.get_action(state_dict, deterministic=True)
                
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_successes.append(info.get('success', False))
        
        self.policy.train()
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'success_rate': np.mean(episode_successes)
        }
    
    def _prepare_state(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare state for policy input"""
        return {
            'image': torch.tensor(obs['image'], dtype=torch.float32, device=self.device).unsqueeze(0).permute(0, 3, 1, 2) / 255.0,
            'instruction': [obs.get('instruction', '')],
            'numeric': torch.tensor(obs['numeric'], dtype=torch.float32, device=self.device).unsqueeze(0)
        }
    
    def save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'episode_count': self.episode_count,
        }, path)
    
    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.episode_count = checkpoint['episode_count']
