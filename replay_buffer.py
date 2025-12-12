import torch 
import gymnasium as gym 
from stable_baselines3 import PPO 
from collections import deque 
import numpy as np 
import pickle

class Buffer():
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer)<self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position+1) % self.capacity
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return (torch.tensor(states, dtype=torch.float32), 
                torch.tensor(actions, dtype=torch.float32), 
                torch.tensor(rewards, dtype=torch.float32), 
                torch.tensor(next_states, dtype=torch.float32), 
                torch.tensor(dones, dtype=torch.float32))
    
    def get_all(self):
        states, actions, rewards, next_states, dones = zip(*self.buffer)
        return (torch.tensor(states, dtype=torch.float32), 
                torch.tensor(actions, dtype=torch.float32), 
                torch.tensor(rewards, dtype=torch.float32), 
                torch.tensor(next_states, dtype=torch.float32), 
                torch.tensor(dones, dtype=torch.float32))
    def __len__(self):
        return len(self.buffer)

def collect_demonstrations(model, env):
    buffer = Buffer()

    for ep in range(100):
        state, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(state, deterministic=True)
            next_state, reward, terminated , truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
        print(f'Ep {ep} done')
    env.close()
    return buffer

def save_buffer(buffer, filename='expert_demos.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(buffer.buffer, f)
    print(f'Buffer saved to {filename}')

if __name__== '__main__':
    env = gym.make('Hopper-v5', render_mode='rgb_array')
    model= PPO.load('./models/final_model.zip')
    buffer = collect_demonstrations(model, env)
    save_buffer(buffer)
