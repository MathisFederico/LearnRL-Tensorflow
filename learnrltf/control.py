import learnrl as rl
import tensorflow as tf


class Control:
    def __init__(self, action_space, exploration):
        self.exploration = exploration
        self.action_space = action_space

    def act(self, Q):
        raise NotImplementedError("You must define 'act(self, Q)' when subclassing Control.")

    def __call__(self, Q, greedy=False):
        if greedy:
            actions = tf.math.argmax(Q, axis=-1)
        else:
            actions = self.act(Q)

        return actions


class Greedy(Control):
    def act(self, Q):
        greedy_actions = tf.math.argmax(Q, axis=-1, output_type=tf.int32)

        action_size = self.action_space.n
        batch_size = Q.shape[0]

        # Disable pylint for next line since there is a bug with tf and pylint
        # pylint: disable=unexpected-keyword-arg
        random_actions = tf.random.uniform(
            (batch_size,), minval=0, maxval=action_size, dtype=tf.int32
        )

        rd = tf.random.uniform((batch_size,), 0, 1)
        actions = tf.where(rd < self.exploration, random_actions, greedy_actions)

        return actions
