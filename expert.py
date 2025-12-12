import torch 
import gymnasium as gym 
from stable_baselines3 import PPO 
from stable_baselines3.common.vec_env import DummyVecEnv 
from stable_baselines3.common.evaluation import evaluate_policy 
from stable_baselines3.common.callbacks import EvalCallback 
import os
import matplotlib.pyplot as plt 
from collections import deque 

env = gym.make('Hopper-v5', render_mode='rgb_array')
obs = env.observation_space.shape[0]
action_space = env.action_space.shape[0]
eval_env = gym.make('Hopper-v5', render_mode='rgb_array')

def train(total_timesteps, log_dir='./logs/', model_dir='./models/'):

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    print('\n=====TRAINING=======')
    
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef = 0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log='./ppo_expert'
    )

    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    save_path = os.path.join(model_dir, 'final_model')
    model.save(save_path)
    print(f'\nExpert Model saved to {save_path}')
    
    
    print("\n=== Evaluating Expert Policy ===")
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=10,
        deterministic=True
    )
    print(f"Expert Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()
    eval_env.close()
    
    return model


if __name__ == '__main__':
    train(total_times=1_000_000)
