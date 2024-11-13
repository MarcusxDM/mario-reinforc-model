from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

## Pre processar env
# Frames e Escala Cinza
from gym.wrappers import GrayScaleObservation
# Wrappers de Vectorizacao
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# Matpotlib
from matplotlib import pyplot as plt

import os
from stable_baselines3 import PPO
# from stable_baselines3.common import BaseCallback


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

# state = env.reset()
# plt.imshow(state[0])
# plt.show() # mostrar primeiro frame

# Criar modelo
# model = PPO('CnnPolicy', env, verbose=1, tensorboard_log='./logs/', learning_rate=0.000001, n_steps=512)

# # Treinar
# model.learn(total_timesteps=5000)

# done = True
# for step in range(5000):
#     if done:
#         state = env.reset()
#     state, reward, done, info = env.step([env.action_space.sample()])
#     env.render()

# env.close()