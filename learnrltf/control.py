import learnrl as rl
import tensorflow as tf


class Control:
    """
    Control base object.

    This method must be specified:
        act(self, Q)

    Parameters
    ----------
        exploration: :class:`float`
            The initial exploration constant.
        name: :class:`str`
            The name of the control.
        exploration_decay: :class:`float`
            The exploration decay, default is 0.

    Attributes
    ----------
        exploration: :class:`float`
            The exploration constant.
        name: :class:`str`
            The name of the control.
        decay: :class:`float`
            The exploration decay.
    """

    def __init__(self, exploration, name, exploration_decay=0):
        if name is None:
            raise ValueError("The Control object must have a name.")

        self.exploration = exploration
        self.name = name
        self.decay = exploration_decay

    def update_exploration(self, exploration=None):
        """
        Update the exploration constant.

        By default, uses exploration_decay to update exploration with formula :math:`exploration *= (1 - decay)`.

        Arguments
        ---------
            exploration: :class:`float`, optional
                The fixed exploration constna that we set to.
        """

        if exploration is not None:
            self.exploration = exploration
        else:
            self.exploration *= 1 - self.decay

    def act(self, Q):
        """
        Return the policy of the agent given the action values.

        Arguments
        ---------
            Q: :class:`tensorflow.Tensor`
                The estimation of Q(s, a), ie the expected future reward if we do action 'a' in state 's'.
                The shape is (sample_size, action_size).

        Return
        ------
            actions: :class:`tensorflow.Tensor`
                Taken actions of shape (sample_size, 1)

        """
        raise NotImplementedError(
            "You must define 'act(self, Q)' when subclassing Control."
        )

    def __call__(self, Q, greedy=False):
        if greedy:
            actions = tf.math.argmax(Q, axis=-1)
        else:
            actions = self.act(Q)

        return actions


class Greedy(Control):
    """
    Greedy control.

    Takes the action maximizing action_value with probability (1 - exploration).
    Else take an uniformly random action.
    """

    def __init__(self, exploration=0.1, **kwargs):
        super().__init__(exploration=exploration, name="greedy", **kwargs)

    def act(self, Q):
        if not 0 <= self.exploration <= 1:
            raise ValueError(
                f"Exploration should be in [0, 1] for greedy control, but was {self.exploration}."
            )

        batch_size, action_size = Q.shape
        greedy_actions = tf.math.argmax(Q, axis=-1, output_type=tf.int32)

        # Disable pylint for next line since there is a bug with tf and pylint
        # pylint: disable=unexpected-keyword-arg
        random_actions = tf.random.uniform(
            (batch_size,), minval=0, maxval=action_size, dtype=tf.int32
        )

        rd = tf.random.uniform((batch_size,), 0, 1)
        actions = tf.where(rd < self.exploration, random_actions, greedy_actions)

        return actions
