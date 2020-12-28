import gym
import learnrl as rl
import learnrltf as rltf

import tensorflow.keras as keras

env = gym.make('CartPole-v1')

action_value = keras.models.Sequential(
    [
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(env.action_space.n, activation='linear')
    ]
)

control = rltf.control.Greedy()
evaluation = rltf.evaluation.QLearning()
memory = rltf.memory.Memory()

agent = rltf.agents.Q.DQAgent(env.observation_space, env.action_space, control, evaluation, action_value, memory)

playground = rl.Playground(env, agent)
playground.fit(500, verbose=2)
playground.test(10)

