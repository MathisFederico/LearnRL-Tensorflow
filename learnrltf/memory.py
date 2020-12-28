import tensorflow as tf

class Memory:
    """
    A general memory for reinforcement learning agents.

    Attributes
    ----------
        max_memory_len: :class:`int`
            Max number of experiences stocked, default is 1000.
        datas: :class:`dict`
            The dictionary of experiences as :class:`tensorflow.Tensor`.
        MEMORY_KEYS:
            | The keys of core parameters to gather from experience
            | ('observation', 'action', 'done', 'next_observation', 'info').
    """

    def __init__(self, max_memory_len=1000):
        self.max_memory_len = max_memory_len
        self.memory_len = 0

        self.MEMORY_KEYS = (
            "observation",
            "action",
            "reward",
            "done",
            "next_observation",
        )
        self.datas = {key: None for key in self.MEMORY_KEYS}

    def remember(self, observation, action, reward, done, next_observation=None):
        """
        Add the nex experience into the memory forgetting long past exprience if necessary.

        Parameters
        ----------
            observation:
                The observation given by the |gym.Env|.
            action:
                The action given to |gym.Env|.
            reward: :class:`float`
                The reward given by the |gym.Env|.
            done:
                Whether the |gym.Env| has ended after the action.
            next_observation:
                The next observation given by the |gym.Env|.
        """

        for val, key in zip(
            (observation, action, reward, done, next_observation), self.MEMORY_KEYS
        ):
            batched_value = tf.expand_dims(val, axis=0)
            if self.memory_len == 0:
                self.datas[key] = batched_value
            else:
                self.datas[key] = tf.concat((self.datas[key], batched_value), axis=0) # pylint: disable=no-value-for-parameter, unexpected-keyword-arg
            self.datas[key] = self.datas[key][-self.max_memory_len :]
        self.memory_len = len(self.datas[self.MEMORY_KEYS[0]])

    def sample(self, sample_size, method="uniform"):
        """
        Return a sample of experiences stored in the memory.

        Parameters
        ----------
            sample_size: :class:`int`
                The size of the sample to get from memory, if 0 return all memory.
            method: :class:`str`
                | The sampling method, default is "uniform".
                | One of ("last", "uniform")
        """

        if method not in ['uniform', 'last']:
            raise NotImplementedError(f"Method {method} is not implemented yet.")

        if self.memory_len <= sample_size or sample_size == 0:
            datas = [self.datas[key] for key in self.MEMORY_KEYS]

        if method == "uniform":
            indices = tf.random.shuffle(tf.range(self.memory_len))[:sample_size]
            datas = [tf.gather(self.datas[key], indices) for key in self.MEMORY_KEYS] # pylint: disable=no-value-for-parameter
        elif method == "last":
            datas = [self.datas[key][-sample_size:] for key in self.MEMORY_KEYS]
        else:
            raise NotImplementedError(f"Unknown sampling method {method}")

        return datas

    def __len__(self):
        return self.memory_len

    def forget(self):
        """Remove all experiences."""

        self.datas = { key: None for key in self.MEMORY_KEYS }
