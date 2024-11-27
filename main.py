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

from stable_baselines3 import PPO, DQN

from train_logs import TrainAndLoggingCallback

CHECKPOINT_PATH='./train/'
LOG_PATH='./logs/'

def train_model(env):
    # Auto Save
    callback = TrainAndLoggingCallback(check_freq=5000, save_path=CHECKPOINT_PATH, model_name='PPO')

    # Criar modelo
    model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_PATH, learning_rate=0.000009, n_steps=512)

    # Treinar
    model.learn(total_timesteps=4000000, callback=callback)
    return True

def train_dqn_model(env):
    # Auto Save
    callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_PATH, model_name='DQN')

    # Criar modelo
    model = DQN('CnnPolicy', env, verbose=1, tensorboard_log=LOG_PATH, learning_rate=1e-4)

    # Treinar
    model.learn(total_timesteps=4000000, callback=callback)
    return True

def load_model(path, model):
    return model.load(path)

def run_model(model_obj, env):
    state = env.reset()
    while True:
        action, _state = model_obj.predict(state)
        state, reward, done, info = env.step(action)
        env.render()

if __name__ == "__main__":

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')

    
    # state = env.reset()
    # plt.imshow(state[0])
    # plt.show() # mostrar primeiro frame


    # train_model(env)
    

    # done = True
    # for step in range(5000):
    #     if done:
    #         state = env.reset()
    #     state, reward, done, info = env.step([env.action_space.sample()])
    #     env.render()

    model = load_model(model=PPO, path='train/best_model_1500000.zip')

    # model.learn(total_timesteps=2000)
    # model.save('best_model_{timesteps}')
    
    run_model(model, env)

    env.close()



