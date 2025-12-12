import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch 
model =torch.load('./models/behaviourC_policy.pth', weights_only=False)
env = gym.make('Hopper-v5', render_mode='rgb_array')

obs, _ = env.reset()
frames = []
done = False
total_reward = 0
while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    frame = env.render()
    frames.append(frame)

env.close()

# Save as gif or video
fig = plt.figure()
im = plt.imshow(frames[0])
plt.axis('off')

def update(frame):
    im.set_array(frame)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
ani.save('bc_hopper.gif', writer='pillow', fps=30)
print("Saved to gail_hopper.gif")
print('total_reward', total_reward)

# TOTAL REWARD: 633.487165738745
