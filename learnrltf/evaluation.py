import learnrl as rl
import tensorflow as tf


class Evaluation:
    """
    Evaluation base object.

    This method must be specified:
        eval(self, rewards, dones, next_observations, action_value)

    Parameters
    ----------
        name: :class:`str`
            Name of the evaluation.
        discount: :class:`float`
            Discount factor, default is 0.99.

    Attributes
    ---------
        All arguments become attributes.
    """

    def __init__(self, name, discount=0.99):
        if name is None:
            raise ValueError("The Evaluation object must have a name.")

        self.name = name
        self.discount = discount

    def eval(self, rewards, dones, next_observations, action_value):
        """
        Evaluate the expected future rewards.

        Arguments
        ---------
            rewards: :class:`tensorflow.Tensor`, float
                The real reward of the last step.
                Shape is (batch_size, ).
            dones: :class:`tensorflow.Tensor`, bool
                True if the environment has ended and the previous step was the last.
                Shape is (batch_size, ).
            next_observations: :class:`tensorflow.Tensor`, float
                The observation made after the step, used to predict what will happen next.
            action_value:
                The estimator used by the agent.
        """

        raise NotImplementedError(
            "You must define"
            " 'eval(self, rewards, dones, next_observations, action_value)'"
            " when subclassing Evaluation."
        )

    def __call__(self, rewards, dones, next_observations, action_value):
        return self.eval(rewards, dones, next_observations, action_value)


class QLearning(Evaluation):
    """
    QLearning is just TemporalDifference with Greedy target control.

    This object is optimized for computation speed.
    """

    def __init__(self, **kwargs):
        super().__init__(name="qlearning", **kwargs)

    def eval(self, rewards, dones, next_observations, action_value):
        future_rewards = rewards

        not_dones = tf.logical_not(dones)
        if tf.math.reduce_any(not_dones):
            Q_max = tf.math.reduce_max(
                action_value(next_observations[not_dones]), axis=-1
            )

            not_dones_indices = tf.where(not_dones)
            future_rewards = tf.tensor_scatter_nd_add(
                future_rewards, not_dones_indices, self.discount * Q_max
            )

        return future_rewards
