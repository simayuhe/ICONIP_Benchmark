# import time
import functools
import tensorflow as tf

# from baselines import logger

# from baselines.common import set_global_seeds, explained_variance
from Common import tf_util
from Common.policies import build_policy


from A2C.utils import Scheduler, find_trainable_variables
# from baselines.a2c.runner import Runner
# from baselines.ppo2.ppo2 import safemean
# from collections import deque

from tensorflow import losses

class A2CAgent():
    def __init__(self,env,args):
        self.num_actions = args.num_actions
        self.num_env = args.num_env
        self.nsteps = args.nsteps
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.max_grad_norm = args.max_grad_norm 
        self.lr = args.lr
        self.alpha = args.alpha
        self.epsilon = args.epsilon
        self.lrschedule = args.lrschedule
        self.network = args.network
        sess = tf_util.get_session()
        nenvs = args.num_env
        nbatch = nenvs*self.nsteps
        policy = build_policy(env, self.network)
        # with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
        with tf.compat.v1.variable_scope('a2c_model', reuse=tf.compat.v1.AUTO_REUSE):#2022年8月1日
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nbatch, self.nsteps, sess)
            
        A = tf.compat.v1.placeholder(train_model.action.dtype, train_model.action.shape)#2022年8月1日
        ADV = tf.compat.v1.placeholder(tf.float32, [nbatch])
        R = tf.compat.v1.placeholder(tf.float32, [nbatch])
        LR = tf.compat.v1.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = tf.compat.v1.losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy*self.ent_coef + vf_loss * self.vf_coef

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if self.max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=LR, decay=self.alpha, epsilon=self.epsilon)

        _train = trainer.apply_gradients(grads)

        self.lrlist = Scheduler(v=self.lr, nvalues=args.total_steps, schedule=self.lrschedule)
        
        def train(obs, states, rewards, masks, actions, values):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = self.lrlist.value()

            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy
        
        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.compat.v1.global_variables_initializer().run(session=sess)
        
