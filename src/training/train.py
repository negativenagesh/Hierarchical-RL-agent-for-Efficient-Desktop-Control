"""Training script"""

import argparse
import torch
from pathlib import Path

from src.agent.policy import HierarchicalPolicy
from src.environment.osworld_wrapper import OSWorldEnv
from src.environment.base_env import TaskDifficulty
from src.training.ppo_trainer import PPOTrainer
from src.training.curriculum import CurriculumManager
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Hierarchical RL Agent")
    
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                       help="Total training timesteps")
    parser.add_argument("--rollout-steps", type=int, default=2048,
                       help="Steps per rollout")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--difficulty", type=str, default="EASY",
                       choices=["EASY", "MEDIUM", "HARD"],
                       help="Task difficulty filter")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda or cpu)")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                       help="Checkpoint save directory")
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Log directory")
    parser.add_argument("--tasks-file", type=str, default="config/tasks.json",
                       help="Tasks JSON file")
    parser.add_argument("--curriculum", action="store_true",
                       help="Enable curriculum learning")
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup logging
    setup_logging(log_dir=args.log_dir)
    logger.info("Starting training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create environment
    difficulty = TaskDifficulty[args.difficulty] if not args.curriculum else None
    env = OSWorldEnv(
        tasks_file=args.tasks_file,
        difficulty_filter=difficulty
    )
    logger.info(f"Environment created with {len(env.tasks)} tasks")
    
    # Create policy
    policy = HierarchicalPolicy()
    logger.info("Policy created")
    
    # Create trainer
    trainer = PPOTrainer(
        policy=policy,
        env=env,
        learning_rate=args.learning_rate,
        device=str(device),
        log_dir=args.log_dir
    )
    logger.info("Trainer created")
    
    # Curriculum learning
    if args.curriculum:
        curriculum = CurriculumManager()
        logger.info("Curriculum learning enabled")
        # TODO: Integrate curriculum with trainer
    
    # Train
    logger.info("Starting training loop")
    trainer.train(
        total_timesteps=args.total_timesteps,
        rollout_steps=args.rollout_steps,
        save_dir=args.save_dir
    )
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
