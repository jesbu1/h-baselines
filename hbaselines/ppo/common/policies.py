import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from baselines.common.tf_util import adjust_shape
from baselines.common.running_mean_std import RunningMeanStd


class PolicyWithValue(object):
    """
    Encapsulates fields and methods for RL policy and value function estimation with shared parameters
    """

    def __init__(self,
                 sess,
                 obs_ph,
                 ac_ph,
                 layers,
                 act_fun,
                 normalize_observations,
                 layer_norm):
        """TODO

        Parameters
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """
        self.sess = sess
        self.X = obs_ph
        self.state = tf.constant([])
        self.initial_state = None

        # =================================================================== #
        # Part 1. Pre-process inputs.                                         #
        # =================================================================== #

        # Potentially normalize the input observations and actions.
        if normalize_observations:
            encoded_x, self.rms = _normalize_clip_observation(obs_ph)
        else:
            encoded_x = obs_ph
        encoded_x = tf.to_float(encoded_x)

        # Flatten the processed observation.
        encoded_x = tf.layers.flatten(encoded_x)

        # =================================================================== #
        # Part 2. Create the actor network.                                   #
        # =================================================================== #

        with tf.variable_scope('pi', reuse=tf.AUTO_REUSE):
            # Create the hidden layers of the actor network.
            latent = encoded_x
            for i, layer in enumerate(layers):
                latent = self._layer(latent, layer, "fc{}".format(i), act_fun,
                                     layer_norm)

            # Create the mean of the actor network.
            mean = self._layer(latent, ac_ph.shape[-1], "pi_out", None, False)

            # Create the logstd of the actor network.
            logstd = tf.get_variable(
                name='pi_logstd',
                shape=(1, ac_ph.shape[-1]),
                initializer=tf.zeros_initializer()
            )

            self.pd = DiagGaussianProbabilityDistribution(mean, logstd)

        # Take an action
        self.action = self.pd.sample()

        # Calculate the neg log of our probability
        self.neglogp = self.pd.neglogp(self.action)

        # =================================================================== #
        # Part 3. Create the critic network(s).                               #
        # =================================================================== #

        with tf.variable_scope("vf", reuse=tf.AUTO_REUSE):
            # Create the hidden layers of the critic network.
            vf_latent = encoded_x
            for i, layer in enumerate(layers):
                vf_latent = self._layer(vf_latent, layer, "fc{}".format(i),
                                        act_fun, layer_norm)

            # Create the output from the critic network.
            self.vf = self._layer(vf_latent, 1, 'vf_out', None, False)
            self.vf = self.vf[:, 0]

    def _evaluate(self, variables, observation, **extra_feed):
        sess = self.sess
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and \
                        inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    @staticmethod
    def _layer(in_val, num_output, name, act_fun, layer_norm):
        """Create a fully connected layer.

        This is here to reduce code duplication.

        Parameters
        ----------
        in_val : tf.Tensor
            input tensor
        num_output : int
            number of outputs from the layer
        name : str
            the name of the layer
        act_fun : tf.nn.* or None
            activation function. None refers to a linear combination

        Returns
        -------
        tf.Tensor
            the output from the new layer
        """
        ret = tf.compat.v1.layers.dense(
            in_val,
            num_output,
            name=name,
            kernel_initializer=slim.variance_scaling_initializer(
                factor=1.0 / 3.0,
                mode='FAN_IN',
                uniform=True
            )
        )
        if layer_norm:
            ret = tf.contrib.layers.layer_norm(
                ret, center=True, scale=True)

        return ret if act_fun is None else act_fun(ret)

    def step(self, obs, **extra_feed):
        """Compute next action(s) given the observation(s).

        Parameters
        ----------
        obs : array_like
            observation data (either single or a batch)

        Returns
        -------
        array_like
            action
        array_like
            value estimate
        array_like
            next state
        array_like
            negative log likelihood of the action under current policy
            parameters) tuple
        """
        a, v, state, neglogp = self._evaluate(
            [self.action, self.vf, self.state, self.neglogp],
            obs,
            **extra_feed
        )
        if state.size == 0:
            state = None
        return a, v, state, neglogp

    def value(self, obs, *args, **kwargs):
        """Compute value estimate(s) given the observation(s).

        Parameters
        ----------
        obs : array_like
            observation data (either single or a batch)

        Returns
        -------
        array_like
            value estimate
        """
        return self._evaluate(self.vf, obs, *args, **kwargs)


def _normalize_clip_observation(x, clip_range=None):
    if clip_range is None:
        clip_range = [-5.0, 5.0]
    rms = RunningMeanStd(shape=x.shape[1:])
    norm_x = tf.clip_by_value((x - rms.mean) / rms.std,
                              min(clip_range), max(clip_range))
    return norm_x, rms


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
        x : array_like
            TODO

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
