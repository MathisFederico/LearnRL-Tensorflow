import learnrl as rl
import tensorflow as tf


class DQAgent(rl.Agent):
    def __init__(
        self,
        observation_space,
        action_space,
        control=None,
        evaluation=None,
        action_value: tf.keras.Model = None,
        memory=None,
        sample_size=32,
        learning_rate=1e-3,
    ):
        self.control = control
        self.evaluation = evaluation
        self.action_value = action_value
        self.action_value_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.memory = memory
        self.sample_size = sample_size

    def act(self, observation, greedy=False):
        observations = tf.expand_dims(observation, axis=0)
        Q = self.action_value(observations)
        action = self.control(Q, greedy)[0]
        return action.numpy()

    def learn(self):

        if len(self.memory) < self.sample_size:
            return {}

        observations, actions, rewards, dones, next_observations = self.memory.sample(
            self.sample_size
        )
        expected_future_rewards = self.evaluation(
            rewards, dones, next_observations, self.action_value
        )

        with tf.GradientTape() as tape:
            Q = self.action_value(observations, training=True)
            action_indices = tf.stack((tf.range(len(actions)), actions), axis=-1)
            Q_actions = tf.gather_nd(Q, action_indices)

            loss = tf.keras.losses.mean_squared_error(
                expected_future_rewards, Q_actions
            )

            grads = tape.gradient(loss, self.action_value.trainable_weights)
            self.action_value_opt.apply_gradients(
                zip(grads, self.action_value.trainable_weights)
            )

        metrics = {
            "loss": loss.numpy(),
            "exploration": self.control.exploration,
            "learning_rate": self.action_value_opt._decayed_lr(tf.float32).numpy(),
        }

        return metrics

    def remember(
        self, observation, action, reward, done, next_observation=None, info={}, **param
    ):
        self.memory.remember(observation, action, reward, done, next_observation)
