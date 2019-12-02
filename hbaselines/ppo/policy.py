import tensorflow as tf
import functools

from hbaselines.ppo.common.policies import PolicyWithValue
# TODO: remove
from baselines.common.tf_util import save_variables, load_variables


class FeedForwardPolicy(object):
    """Feed-forward policy object.

    This policy is compatible with the PPO training algorithm.

    Attributes
    ----------
    sess : tf.compat.v1.Session
        the tensorflow session
    obs_ph : tf.compat.v1.placeholder
        placeholder for the observations
    ac_ph : tf.compat.v1.placeholder
        placeholder for the actions
    adv_ph : tf.compat.v1.placeholder
        placeholder for the advantages
    ret_ph : tf.compat.v1.placeholder
        placeholder for the returns
    old_neglogpac : tf.compat.v1.placeholder
        placeholder for the negative log-probability of actions in the previous
        step policy
    old_vpred : tf.compat.v1.placeholder
        placeholder for the predicted values from the previous step policy
    learning_rate : tf.compat.v1.placeholder
        placeholder for the current learning rate
    clip_range : tf.compat.v1.placeholder
        placeholder for the current clip range to the gradients
    trainer : TODO
        TODO
    grads : TODO
        TODO
    var : TODO
        TODO
    _train_op : tf.Operation
        the training operation
    loss_names : list of str
        the names of the individual losses; used for logging purposes
    stats_list : list of tf.Operation
        the tensorflow operations that produce the individual elements of the
        loss; used for logging purposes
    train_model : TODO
        TODO
    act_model : TODO
        TODO
    step : TODO
        TODO
    value : TODO
        TODO
    """

    def __init__(self,
                 *,
                 sess,
                 policy,
                 ob_space,
                 ac_space,
                 nbatch_act,
                 nbatch_train,
                 nsteps,
                 ent_coef,
                 vf_coef,
                 max_grad_norm,
                 microbatch_size=None):
        """Instantiate the policy object.

        Parameters
        ----------
        ob_space : gym.spaces.*
            the observation space of the environment
        ac_space : gym.spaces.*
            the action space of the environment
            number of training steps per iteration
        ent_coef : float
            policy entropy coefficient in the optimization objective
        vf_coef : float
            value function loss coefficient in the optimization objective
        max_grad_norm : float or None
            gradient norm clipping coefficient
        """
        self.sess = sess

        # =================================================================== #
        # Part 1. Create the placeholders.                                    #
        # =================================================================== #

        self.obs_ph = tf.placeholder(tf.float32, [None, ob_space.shape[0]])
        self.ac_ph = tf.placeholder(tf.float32, [None, ac_space.shape[0]])
        self.adv_ph = tf.placeholder(tf.float32, [None])
        self.ret_ph = tf.placeholder(tf.float32, [None])
        self.old_neglogpac = tf.placeholder(tf.float32, [None])
        self.old_vpred = tf.placeholder(tf.float32, [None])
        self.learning_rate = tf.placeholder(tf.float32, [])
        self.clip_range = tf.placeholder(tf.float32, [])

        # =================================================================== #
        # Part 2. Create the policies.                                        #
        # =================================================================== #

        with tf.variable_scope('ppo2_model', reuse=tf.AUTO_REUSE):
            # act_model that is used for sampling
            act_model = policy(nbatch_act, 1, sess)

            # Train model for training
            if microbatch_size is None:
                train_model = policy(nbatch_train, nsteps, sess)
            else:
                train_model = policy(microbatch_size, nsteps, sess)

        # =================================================================== #
        # Part 3. Calculate the loss.                                         #
        #                                                                     #
        # Total loss = policy gradient loss - entropy * entropy coefficient   #
        #              + Value coefficient * value loss                       #
        # =================================================================== #

        neglogpac = train_model.pd.neglogp(self.ac_ph)

        # Calculate the entropy. Entropy is used to improve exploration by
        # limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = self.old_vpred + tf.clip_by_value(
            train_model.vf - self.old_vpred,
            - self.clip_range, self.clip_range)
        # Unclipped value
        vf_losses1 = tf.square(vpred - self.ret_ph)
        # Clipped value
        vf_losses2 = tf.square(vpredclipped - self.ret_ph)

        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

        # Calculate ratio (pi current policy / pi old policy)
        ratio = tf.exp(self.old_neglogpac - neglogpac)

        # Defining Loss = - J is equivalent to max J
        pg_losses = -self.adv_ph * ratio

        pg_losses2 = -self.adv_ph * tf.clip_by_value(
            ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)

        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(
            tf.square(neglogpac - self.old_neglogpac))
        clipfrac = tf.reduce_mean(
            tf.to_float(tf.greater(tf.abs(ratio - 1.0), self.clip_range)))

        # Total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        # =================================================================== #
        # Part 4. Create the parameter update procedure.                      #
        # =================================================================== #

        # 1. Get the model parameters
        params = tf.trainable_variables('ppo2_model')

        # 2. Build our trainer
        self.trainer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, epsilon=1e-5)

        # 3. Calculate the gradients
        grads_and_var = self.trainer.compute_gradients(loss, params)
        grads, var = zip(*grads_and_var)

        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da
        grads_and_var = list(zip(grads, var))

        self.grads = grads
        self.var = var
        self._train_op = self.trainer.apply_gradients(grads_and_var)
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy',
                           'approxkl', 'clipfrac']
        self.stats_list = [pg_loss, vf_loss, entropy, approxkl, clipfrac]

        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step  # TODO: remove
        self.value = act_model.value
        self.initial_state = act_model.initial_state  # TODO: remove

        # TODO: move to separate methods
        self.save = functools.partial(save_variables, sess=sess)
        self.load = functools.partial(load_variables, sess=sess)

        # =================================================================== #
        # Part 5. Initialize all parameter.                                   #
        # =================================================================== #

        self.sess.run(tf.global_variables_initializer())

    def train(self,
              lr,
              clip_range,
              obs,
              returns,
              masks,
              actions,
              values,
              neglogpacs,
              states=None):
        """TODO

        Parameters
        ----------
        lr : float
            the current learning rate
        clip_range : float
            the current clip range for the gradients
        obs : array_like
            (batch_size, obs_dim) matrix of observations
        returns : array_like
            (batch_size,) vector of returns
        actions : array_like
            (batch_size, ac_dim) matrix of actions
        values : array_like
            (batch_size,) vector of values
        neglogpacs : array_like
            negative log probabilities of the old policy TODO size

        Returns
        -------
        list of float  TODO: array_like? try printing
            the values associated with the individual elements of the loss
            function
        """
        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advs = returns - values

        # Normalize the advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        td_map = {
            self.obs_ph: obs,
            self.ac_ph: actions,
            self.adv_ph: advs,
            self.ret_ph: returns,
            self.learning_rate: lr,
            self.clip_range: clip_range,
            self.old_neglogpac: neglogpacs,
            self.old_vpred: values
        }
        if states is not None:
            td_map[self.train_model.S] = states
            td_map[self.train_model.M] = masks

        return self.sess.run(self.stats_list + [self._train_op], td_map)[:-1]
