import learnrl as rl
import tensorflow as tf

class Evaluation():

    def __init__(self, discount=0.99):
        self.discount = discount

    def eval(self, rewards, dones, next_observations, action_value):
        raise NotImplementedError("You must define"
                                  " 'eval(self, rewards, dones, next_observations, action_value)'"
                                  " when subclassing Evaluation.")

    def __call__(self, rewards, dones, next_observations, action_value):
        return self.eval(rewards, dones, next_observations, action_value)

class QLearning(Evaluation):

    def eval(self, rewards, dones, next_observations, action_value):
        future_rewards = rewards

        not_dones = tf.logical_not(dones)
        if tf.math.reduce_any(not_dones):
            Q_max = tf.math.reduce_max(action_value(next_observations[not_dones]), axis=-1)

            not_dones_indices = tf.where(not_dones)
            future_rewards = tf.tensor_scatter_nd_add(future_rewards, not_dones_indices, self.discount * Q_max)

        return future_rewards
