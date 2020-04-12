
import gym #type:ignore
import tensorflow as tf #type:ignore
import numpy as np #type:ignore
from collections import deque
import time
from typing import  List

class ReplayBuffer:

    def __init__(self, buffer_size):
        self.replay_buffer = []
        self.buffer_size = buffer_size

    def sample_experience(self, batch_size):
        max_mem = min(self.buffer_size, len(self.replay_buffer))
        indices = np.random.randint(max_mem, size = batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        states, actions, rewards, states_, dones = [np.array([experience[field_index] for experience in batch]) for field_index in range(5)]
        return (states, actions, rewards, states_, dones)



class Net:

    def __init__(self, input_shape, num_actions, learning_rate, batch_size, epsilon, eps_decay, discount_factor):

        self.input_ = tf.keras.layers.Input(shape = input_shape)
        self.hidden1 = tf.keras.layers.Dense(256, activation="relu")(self.input_)
        self.hidden2 = tf.keras.layers.Dense(256, activation="relu")(self.hidden1)
        self.output = tf.keras.layers.Dense(num_actions)(self.hidden2)
        self.learning_rate = learning_rate
        self.replay_buffer = ReplayBuffer(1000000)
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.model = tf.keras.Model(inputs = [self.input_], outputs=[self.output])
        self.model.compile(optimizer = tf.keras.optimizers.Adam(lr = self.learning_rate),loss = "mean_squared_error")

    def train(self):
        states, actions, rewards, states_, dones = self.replay_buffer.sample_experience(self.batch_size)
        # print(states.shape)
        # print(states)
        # print(states_.shape)

        # print(states.shape)
        predicted = self.model.predict(states)
        # print(predicted.shape)
        # print(rewards.shape)
        # rewards = rewards.reshape(-1,1)
        actual = np.copy(predicted)
        # dones = dones.reshape(-1,1)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # print(f'bi:{batch_index.shape}')
        # print(f'act:{actions.shape}')
        actual[batch_index, actions] = rewards + self.discount_factor*np.max(self.model.predict(states_), axis=1)*(1-dones)#.reshape(-1,1)
        # print('np.max:',np.max(self.model.predict(states_), axis=1,keepdims=True)*(1-dones))
        # print(actual.shape)
        self.model.fit(states, actual, verbose=0, epochs=1)
        self.epsilon = self.epsilon - self.eps_decay if self.epsilon>=0.01 else 0.01

    def policy(self, state):

        # print(state)

        if np.random.random()<self.epsilon:
            #do something random
            return np.random.randint(0,self.num_actions)
        else:
            # print(state.shape)
            # print(self.model.predict(state))
            actions = self.model.predict(state[np.newaxis])
            return np.argmax(actions)



if __name__ == '__main__':
    dqn_agent = Net([8], 4, 0.001, 64, 0.98, 0.001, 0.99)

    NUM_EPISODES = 1000
    MAX_STEPS = 1000

    env = gym.make('LunarLander-v2')
    print(env.observation_space.shape)
    print(env.action_space.n)

    print(dqn_agent.model.summary())
    rewards:List[float]=[]
    best_running_avg = -np.inf
    for episode in range(NUM_EPISODES):
        print(episode)
        reward_total = 0
        t=0
        obs = env.reset()
        # print(obs.shape)
        done = False

        while not done:
            action = dqn_agent.policy(obs)
            obs_, reward, done, info = env.step(action)
            dqn_agent.replay_buffer.replay_buffer.append((obs, action, reward, obs_, done))
            obs = obs_
            t+=1
            reward_total=reward_total+reward
            if episode%10 == 0:
                time.sleep(0.001)
                env.render()

            if episode > 2:
                dqn_agent.train()

            if done:
                rewards.append(reward_total)
                print(f'Episode{episode} finished after {t} timesteps with running average {np.mean(rewards[-10:])} reward with epsilon {dqn_agent.epsilon}')
                if np.mean(rewards[-10:]) > best_running_avg:
                    best_running_avg = np.mean(rewards[-10:])
                    print(best_running_avg)
                    tf.keras.models.save_model(dqn_agent.model, f'dqn-lunar-lander.h5')




    env.close()
    # tf.keras.models.save_model(dqn_agent.model, 'dqn.h5')
