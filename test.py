import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Load the trained GAIL model
model = PPO.load('./models/gail_trained_model')

# Create environment
env = gym.make('Hopper-v5', render_mode='rgb_array')  # 'human' to visualize

# Evaluate the model
print("Evaluating GAIL trained model...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

env.close()
# Mean reward: 150.20 +/- 7.47
