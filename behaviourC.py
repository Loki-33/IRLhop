import torch 
import torch.nn as nn 
from replay_buffer import Buffer 
import pickle
import numpy as np
import gymnasium as gym 
from imitation.util.networks import RunningNorm
from imitation.algorithms.adversarial import gail
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.algorithms import bc
from imitation.data import types
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

def generate_expert_trajectories(expert_model_path, n_episodes=50):
    """Generate expert trajectories from saved model"""
    # Load expert
    expert = PPO.load(expert_model_path)
    
    # Create environment
    venv = DummyVecEnv([make_env()])
    
    # Generate trajectories using imitation's rollout
    trajectories = rollout.rollout(
        expert,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=n_episodes),
        rng=np.random.default_rng(42)
    )
    
    print(f"Generated {len(trajectories)} expert trajectories")
    return trajectories

    
def make_env():
    def _init():
        env = gym.make('Hopper-v5', render_mode='rgb_array')
        env = RolloutInfoWrapper(env)
        return env 
    return _init
    
def train_bc(expert_trajectories):
    print("===============TRAINING BC===================")
    venv = DummyVecEnv([make_env()])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False)

    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        demonstrations=expert_trajectories,
        rng=np.random.default_rng(42),
        batch_size=128,
        ent_weight=1e-3,
        l2_weight=1e-5
    )

    print('starting BC....')
    bc_trainer.train(
        n_epochs=200,
        log_interval=10,
        progress_bar=True
    )

    print('\bTRAINING COMPLETE...')
    return bc_trainer, venv


if __name__ == '__main__':
    env = gym.make('Hopper-v5', render_mode='rgb_array')
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    expert_trajectories = generate_expert_trajectories('./models/final_model', n_episodes=50)
    bc_trainer, venv = train_bc(expert_trajectories)
    bc_policy = bc_trainer.policy
    torch.save(bc_policy, './models/behaviourC_policy.pth')
    print("Model saved!!!!!!!!!")
