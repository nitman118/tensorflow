import tensorflow as tf #type:ignore
from dqn_lunar_lander import Net
import gym #type:ignore
import time
import this #type:ignore
env = gym.make('MountainCar-v0')

NUM_EPISODES = 20

# dqn_agent = Net([8], 4, 0.001, 64, 0.000001, 0.001, 0.99)
dqn_agent = Net([2], 3, 0.003, 32,0, 0.001, 0.9995)

dqn_agent.model = tf.keras.models.load_model('trained-policies/dqn-mountain-car.h5')

for episode in range(NUM_EPISODES):
    print(episode)

    obs = env.reset()
    # print(obs.shape)
    done = False

    while not done:
        action = dqn_agent.policy(obs)
        # action = env.action_space.sample()
        obs_, reward, done, info = env.step(action)
        # dqn_agent.replay_buffer.replay_buffer.append((obs, action, reward, obs_, done))
        obs = obs_
        time.sleep(0.01)
        env.render()

env.close()
