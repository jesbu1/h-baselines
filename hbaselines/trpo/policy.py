import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from stable_baselines.common.policies import ActorCriticPolicy


class FeedForwardPolicy(object):
    """Actor critic feed forward neural network.

    Attributes
    ----------
    sess : tf.Session
        The current TensorFlow session
    ob_space : gym.spaces.*
        The observation space of the environment
    ac_space : gym.spaces.*
        The action space of the environment
    reuse : bool
        If the policy is reusable or not
    layers : list of int
        The size of the Neural network for the policy (if None, default to
        [64, 64])
    act_fun : tf.nn.*
        the activation function to use in the neural network.
    obs_ph : tf.compat.v1.placeholder
        observation placeholder
    value_fn : tf.Tensor
        value estimate, of shape (self.n_batch, 1)
    value_flat : tf.Tensor
        value estimate, of shape (self.n_batch, )
    proba_distribution : ProbabilityDistribution
        distribution of stochastic actions.
    q_value : tf.Tensor
        TODO
    action : tf.Tensor
        stochastic action, of shape (self.n_batch, ) + self.ac_space.shape.
    deterministic_action : tf.Tensor
        deterministic action, of shape (self.n_batch, ) + self.ac_space.shape.
    neglogp : tf.Tensor
        negative log likelihood of the action sampled by self.action.
    policy_proba : tf.Tensor
        parameters of the probability distribution. Depends on pdtype.
    """

    def __init__(self,
                 sess,
                 ob_space,
                 ac_space,
                 reuse=False,
                 layers=None,
                 act_fun=tf.tanh):
        """Instantiate the policy.

        Parameters
        ----------
        sess : tf.Session
            The current TensorFlow session
        ob_space : gym.spaces.*
            The observation space of the environment
        ac_space : gym.spaces.*
            The action space of the environment
        reuse : bool
            If the policy is reusable or not
        layers : list of int
            The size of the Neural network for the policy (if None, default to
            [64, 64])
        act_fun : tf.nn.*
            the activation function to use in the neural network.
        """
        self.sess = sess
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.reuse = reuse
        self.layers = layers or [64, 64]
        self.act_fun = act_fun
        print(ac_space, ob_space)

        # Create the observation placeholder
        with tf.variable_scope("input", reuse=False):
            self.obs_ph = tf.compat.v1.placeholder(
                tf.float32, shape=(None,) + ob_space.shape)

        with tf.variable_scope("model", reuse=reuse):
            pi_latent, vf_latent = self.mlp_extractor(
                tf.layers.flatten(self.obs_ph), self.layers, self.act_fun)

            # add an extra layer to the shared layers to produce the output
            # from the value function
            self.value_fn = tf.compat.v1.layers.dense(
                vf_latent,
                1,
                name='vf',
                kernel_initializer=slim.variance_scaling_initializer(
                    factor=1.0 / 3.0, mode='FAN_IN', uniform=True),
            )
            self.value_flat = self.value_fn[:, 0]

            # Compute the mean value of the action probability.
            self._mean = tf.compat.v1.layers.dense(
                pi_latent,
                self.ac_space.shape[0],
                name='pi',
                kernel_initializer=slim.variance_scaling_initializer(
                    factor=1.0 / 3.0, mode='FAN_IN', uniform=True),
            )

            # The logstd is a single trainable variable.
            self._logstd = tf.get_variable(
                name='pi/logstd', shape=(1, self.ac_space.shape[0]),
                initializer=tf.zeros_initializer()
            )
            self._std = tf.exp(self._logstd)

            # TODO: remove
            self.proba_distribution = DiagGaussianProbabilityDistribution(
                self._mean, self._logstd
            )

            # Q-function associated with a given action.
            self.q_value = tf.compat.v1.layers.dense(
                vf_latent,
                self.ac_space.shape[0],
                name='q',
                kernel_initializer=slim.variance_scaling_initializer(
                    factor=1.0 / 3.0, mode='FAN_IN', uniform=True),
            )

        # Compute an action from the probability distribution.
        self.action = self._mean + self._std \
            * tf.random_normal(tf.shape(self._mean), dtype=self._mean.dtype)

        # Set up the distributions, actions, and value.
        self.deterministic_action = self._mean
        self.neglogp = self.proba_distribution.neglogp(self.action)
        self.policy_proba = [self._mean, self._std]

    def step(self, obs, deterministic=False):
        """Return the policy for a single step.

        Parameters
        ----------
        obs : array_like
            The current observation of the environment
        deterministic : bool
            Whether or not to return deterministic actions.

        Returns
        -------
        (array_like, array_like, array_like)
            actions, values, neglogp
        """
        if deterministic:
            action, value, neglogp = self.sess.run(
                [self.deterministic_action, self.value_flat, self.neglogp],
                feed_dict={self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run(
                [self.action, self.value_flat, self.neglogp],
                feed_dict={self.obs_ph: obs})

        return action, value, neglogp

    def proba_step(self, obs):
        """Return the action probability for a single step.

        Parameters
        ----------
        obs : array_like
            The current observation of the environment

        Returns
        -------
        array_like
            the action probability
        """
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs):
        """Return the value for a single step.

        Parameters
        ----------
        obs : array_like
            The current observation of the environment

        Returns
        -------
        array_like
            The associated value of the action
        """
        return self.sess.run(self.value_flat, {self.obs_ph: obs})

    @staticmethod
    def mlp_extractor(flat_observations, layers, act_fun):
        """Create the shared layers of the policy.

        These layers are used by both the actor and value function.

        Parameters
        ----------
        flat_observations : tf.Tensor
            The observations to base policy and value function on.
        layers : list of int
            The number of units per layer.
        act_fun : tf.nn.*
            The activation function to use for the networks.

        Returns
        -------
        tf.Tensor
            latent_policy of the specified network
        tf.Tensor
            latent_value of the specified network
        """
        latent = flat_observations

        # Iterate through the shared layers and build the shared parts of the
        # network
        for idx, layer in enumerate(layers):
            layer_size = layer
            latent = tf.compat.v1.layers.dense(
                latent,
                layer_size,
                activation=act_fun,
                name="shared_fc{}".format(idx),
                kernel_initializer=slim.variance_scaling_initializer(
                    factor=1.0 / 3.0, mode='FAN_IN', uniform=True),
            )

        # Build the non-shared part of the network
        latent_policy = latent
        latent_value = latent

        return latent_policy, latent_value


class DiagGaussianProbabilityDistribution(object):
    """TODO

    """

    def __init__(self, mean, logstd):
        """
        Probability distributions from multivariate gaussian input

        :param flat: ([float]) the multivariate gaussian input data
        """
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
        self.flat = tf.concat([mean, mean * 0.0 + logstd], axis=1)

    def flatparam(self):
        """Return the direct probabilities."""
        return self.flat

    def mode(self):
        """Return the probability (deterministic action)."""
        return self.mean

    def neglogp(self, x):
        """Return the of the negative log likelihood.

        Parameters
        ----------
        x : str
            the labels of each index

        Returns
        -------
        array_like
            The negative log likelihood of the distribution
        """
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std),
                                   axis=-1) \
            + 0.5 * np.log(2. * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
            + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        """Calculate the Kullback-Leibler divergence of the distribution.

        Parameters
        ----------
        other : list of float
            the distribution to compare with

        Returns
        -------
        float
            the KL divergence of the two distributions
        """
        assert isinstance(other, DiagGaussianProbabilityDistribution)
        return tf.reduce_sum(
            other.logstd - self.logstd +
            (tf.square(self.std) + tf.square(self.mean - other.mean)) /
            (2.0 * tf.square(other.std)) - 0.5,
            axis=-1)

    def entropy(self):
        """Return shannon's entropy of the probability."""
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e),
                             axis=-1)

    def sample(self):
        """Return a sample from the probability distribution.

        Note: Bounds are taken into account outside this class (during training
        only).

        Returns
        -------
        tf.Tensor
            the stochastic action
        """
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean),
                                                       dtype=self.mean.dtype)

    def logp(self, x):
        """Return the of the log likelihood.

        Parameters
        ----------
        x : str
            the labels of each index

        Returns
        -------
        list of float
            the log likelihood of the distribution
        """
        return - self.neglogp(x)
