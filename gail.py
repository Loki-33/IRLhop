import numpy as np
import torch
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
from stable_baselines3.common.evaluation import evaluate_policy  # YOU FORGOT THIS IMPORT
import gymnasium as gym
import mujoco 
import mujoco.viewer

SEED = 42  

def make_env():
    def _init():
        env = gym.make('Hopper-v4', render_mode='rgb_array')
        env = RolloutInfoWrapper(env)
        return env 
    return _init

def train(total_timesteps=100000):
    venv = DummyVecEnv([make_env()])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    learner = PPO(
        policy='MlpPolicy',
        env=venv,
        batch_size=64,
        n_steps=512,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        gae_lambda=0.95,
        verbose=1, 
        tensorboard_log='./logs',
        device='cpu',
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        )
    )
    
    # Create reward network
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm, 
        hid_sizes=[128, 128]
    )
    
    print('\nGENERATING ROLLOUTS')
    expert = PPO.load('./models/final_model')
    expert_env = DummyVecEnv([make_env()])  
    
    rollouts = rollout.rollout(
        expert, 
        expert_env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),  # YOU FORGOT min_timesteps= AND SET IT TO None
        rng=np.random.default_rng(SEED)
    )
    print(f"Generated {len(rollouts)} expert trajectories")
    
    print('\nINITIALIZING GAIL TRAINER')  
    trainer = gail.GAIL(
        demonstrations=rollouts,
        demo_batch_size=32,
        gen_replay_buffer_capacity=20000,
        n_disc_updates_per_round=2,
        venv=venv,
        gen_algo=learner, 
        reward_net=reward_net,
        allow_variable_horizon=True,
        disc_opt_cls=torch.optim.Adam,
        disc_opt_kwargs={
            'lr': 1e-4,
        }
    )
    print('\nSTARTING TRAINING')
    trainer.train(
        total_timesteps=total_timesteps
    )
    
    print('\nEVALUATING POLICY')
    eval_env = gym.make('Hopper-v4', render_mode='rgb_array')
    mean_reward, std_reward = evaluate_policy(learner, eval_env, n_eval_episodes=10)  # USE learner NOT policy
    print(f"Trained GAIL mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
   
    expert_env.close()
    eval_env.close()
    
    return trainer, learner, venv 

if __name__ == '__main__':
    trainer, learner, venv = train(total_timesteps=2_000_000)  
    

    learner.save('./models/gail_trained_model')
    print('SAVED MODEL')
    
    venv.close()
    print('DONE MF!')
