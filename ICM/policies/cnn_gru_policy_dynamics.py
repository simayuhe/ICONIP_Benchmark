import numpy as np
import tensorflow as tf
# from baselines import logger
from RND.utils import fc, conv
from ICM.stochastic_policy import StochasticPolicy
from RND.tf_util import get_available_gpus
from RND.mpi_util import RunningMeanStd
from ICM.constants import constants

def to2d(x):
    size = 1
    for shapel in x.get_shape()[1:]: size *= shapel.value
    return tf.reshape(x, (-1, size))



class GRUCell(tf.nn.rnn_cell.RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""
    def __init__(self, num_units, rec_gate_init=-1.0):
        tf.nn.rnn_cell.RNNCell.__init__(self)
        self._num_units = num_units
        self.rec_gate_init = rec_gate_init
    @property
    def state_size(self):
        return self._num_units
    @property
    def output_size(self):
        return self._num_units
    def call(self, inputs, state):
        """Gated recurrent unit (GRU) with nunits cells."""
        x, new = inputs
        h = state
        h *= (1.0 - new)
        hx = tf.concat([h, x], axis=1)
        mr = tf.sigmoid(fc(hx, nh=self._num_units * 2, scope='mr', init_bias=self.rec_gate_init))
        # r: read strength. m: 'member strength
        m, r = tf.split(mr, 2, axis=1)
        rh_x = tf.concat([r * h, x], axis=1)
        htil = tf.tanh(fc(rh_x, nh=self._num_units, scope='htil'))
        h = m * h + (1.0 - m) * htil
        return h, h

class CnnGruPolicy(StochasticPolicy):
    def __init__(self, scope, ob_space, ac_space,
                 policy_size='normal', maxpool=False, extrahid=True, hidsize=128, memsize=128, rec_gate_init=0.0,
                 update_ob_stats_independently_per_gpu=True,
                 proportion_of_exp_used_for_predictor_update=1.,
                 dynamics_bonus = False,
                 ):
        StochasticPolicy.__init__(self, scope, ob_space, ac_space)
        print("ac_space.n in cnn gru-------------------:",ac_space.n)
        self.proportion_of_exp_used_for_predictor_update = proportion_of_exp_used_for_predictor_update
        enlargement = {
            'small': 1,
            'normal': 2,
            'large': 4
        }[policy_size]
        rep_size = 512
        self.ph_mean = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape[:2])+[1], name="obmean")
        self.ph_std = tf.placeholder(dtype=tf.float32, shape=list(ob_space.shape[:2])+[1], name="obstd")
        memsize *= enlargement
        hidsize *= enlargement
        convfeat = 16*enlargement
        self.ob_rms = RunningMeanStd(shape=list(ob_space.shape[:2])+[1], use_mpi=not update_ob_stats_independently_per_gpu)
        ph_istate = tf.placeholder(dtype=tf.float32,shape=(None,memsize), name='state')
        pdparamsize = self.pdtype.param_shape()[0]
        self.memsize = memsize

        self.pdparam_opt, self.vpred_int_opt, self.vpred_ext_opt, self.snext_opt = \
            self.apply_policy(self.ph_ob[None][:,:-1],
                              ph_new=self.ph_new,
                              ph_istate=ph_istate,
                              reuse=False,
                              scope=scope,
                              hidsize=hidsize,
                              memsize=memsize,
                              extrahid=extrahid,
                              sy_nenvs=self.sy_nenvs,
                              sy_nsteps=self.sy_nsteps - 1,
                              pdparamsize=pdparamsize,
                              rec_gate_init=rec_gate_init
                              )
        self.pdparam_rollout, self.vpred_int_rollout, self.vpred_ext_rollout, self.snext_rollout = \
            self.apply_policy(self.ph_ob[None],
                              ph_new=self.ph_new,
                              ph_istate=ph_istate,
                              reuse=True,
                              scope=scope,
                              hidsize=hidsize,
                              memsize=memsize,
                              extrahid=extrahid,
                              sy_nenvs=self.sy_nenvs,
                              sy_nsteps=self.sy_nsteps,
                              pdparamsize=pdparamsize,
                              rec_gate_init=rec_gate_init
                              )
        # if dynamics_bonus:
        #     self.define_dynamics_prediction_rew(convfeat=convfeat, rep_size=rep_size, enlargement=enlargement)
        # else:
        #     self.define_self_prediction_rew(convfeat=convfeat, rep_size=rep_size, enlargement=enlargement)
        self.define_state_action_predictor(ob_space=ob_space,ac_space=ac_space)


        pd = self.pdtype.pdfromflat(self.pdparam_rollout)
        self.a_samp = pd.sample()
        self.nlp_samp = pd.neglogp(self.a_samp)
        self.entropy_rollout = pd.entropy()
        self.pd_rollout = pd

        self.pd_opt = self.pdtype.pdfromflat(self.pdparam_opt)

        self.ph_istate = ph_istate

    @staticmethod
    def apply_policy(ph_ob, ph_new, ph_istate, reuse, scope, hidsize, memsize, extrahid, sy_nenvs, sy_nsteps, pdparamsize, rec_gate_init):
        data_format = 'NHWC'
        ph = ph_ob
        assert len(ph.shape.as_list()) == 5  # B,T,H,W,C
        #logger.info("CnnGruPolicy: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
        X = tf.cast(ph, tf.float32) / 255.
        X = tf.reshape(X, (-1, *ph.shape.as_list()[-3:]))

        activ = tf.nn.relu
        yes_gpu = any(get_available_gpus())

        with tf.variable_scope(scope, reuse=reuse), tf.device('/gpu:0' if yes_gpu else '/cpu:0'):
            X = activ(conv(X, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2), data_format=data_format))
            X = activ(conv(X, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2), data_format=data_format))
            X = activ(conv(X, 'c3', nf=64, rf=4, stride=1, init_scale=np.sqrt(2), data_format=data_format))
            X = to2d(X)
            X = activ(fc(X, 'fc1', nh=hidsize, init_scale=np.sqrt(2)))
            X = tf.reshape(X, [sy_nenvs, sy_nsteps, hidsize])
            X, snext = tf.nn.dynamic_rnn(
                GRUCell(memsize, rec_gate_init=rec_gate_init), (X, ph_new[:,:,None]),
                dtype=tf.float32, time_major=False, initial_state=ph_istate)
            X = tf.reshape(X, (-1, memsize))
            Xtout = X
            if extrahid:
                Xtout = X + activ(fc(Xtout, 'fc2val', nh=memsize, init_scale=0.1))
                X = X + activ(fc(X, 'fc2act', nh=memsize, init_scale=0.1))
            pdparam = fc(X, 'pd', nh=pdparamsize, init_scale=0.01)
            vpred_int = fc(Xtout, 'vf_int', nh=1, init_scale=0.01)
            vpred_ext = fc(Xtout, 'vf_ext', nh=1, init_scale=0.01)

            pdparam = tf.reshape(pdparam, (sy_nenvs, sy_nsteps, pdparamsize))
            vpred_int = tf.reshape(vpred_int, (sy_nenvs, sy_nsteps))
            vpred_ext = tf.reshape(vpred_ext, (sy_nenvs, sy_nsteps))
        return pdparam, vpred_int, vpred_ext, snext


    def flatten(self,x):
        return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])

 
    def normalized_columns_initializer(self,std=1.0):
        def _initializer(shape, dtype=None, partition_info=None):
            out = np.random.randn(*shape).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            return tf.constant(out)
        return _initializer

   
    def conv2d(self,x, num_filters, name, filter_size=(3, 3), stride=(1, 1), pad="SAME", dtype=tf.float32, collections=None):
        with tf.variable_scope(name):
            stride_shape = [1, stride[0], stride[1], 1]
            filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = np.prod(filter_shape[:3])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = np.prod(filter_shape[:2]) * num_filters
            # initialize weights with random weights
            w_bound = np.sqrt(6. / (fan_in + fan_out))

            w = tf.get_variable("W", filter_shape, dtype, tf.random_uniform_initializer(-w_bound, w_bound),
                                collections=collections)
            b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.constant_initializer(0.0),
                                collections=collections)
            return tf.nn.conv2d(x, w, stride_shape, pad) + b


    def linear(self,x, size, name, initializer=None, bias_init=0):
        w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=initializer)
        b = tf.get_variable(name + "/b", [size], initializer=tf.constant_initializer(bias_init))
        return tf.matmul(x, w) + b


    def natureHead(self,x):
        ''' DQN Nature 2015 paper
            input: [None, 84, 84, 4]; output: [None, 3136] -> [None, 512];
        '''
        print('Using nature head design')
        x = tf.nn.relu(self.conv2d(x, 32, "l1", [8, 8], [4, 4], pad="VALID"))
        x = tf.nn.relu(self.conv2d(x, 64, "l2", [4, 4], [2, 2], pad="VALID"))
        x = tf.nn.relu(self.conv2d(x, 64, "l3", [3, 3], [1, 1], pad="VALID"))
        x = self.flatten(x)
        x = tf.nn.relu(self.linear(x, 512, "fc", self.normalized_columns_initializer(0.01)))
        return x


    def define_state_action_predictor(self,ob_space=None,ac_space=None):
        # 仿照StateActionPredictor
        # place holder 在Stochastic中定义过了，self.ph_ob ；self.ph_ob_next ；self.ph_ac
        self.s1 =  self.ph_ob
        self.s2 =  self.ph_ob_next
        self.asample = asample = self.ph_ac
        size = 256

        for phi1 in self.ph_ob.values():
            print("phi1 shape ========== :",phi1.shape)
            xr1 = phi1[:,1:]
            xr1 = tf.cast(xr1, tf.float32)
            xr1 = tf.reshape(xr1, (-1, *phi1.shape.as_list()[-3:]))[:, :, :, -1:]
            xr1 = tf.clip_by_value((xr1 - self.ph_mean) / self.ph_std, -5.0, 5.0)
            print("xr1 shape ========== :",xr1.shape)
        for phi2 in self.ph_ob_next.values():
            print("phi2 shape ========== :",phi2.shape)
            xr2 = phi2[:,1:]
            xr2 = tf.cast(xr2, tf.float32)
            xr2 = tf.reshape(xr2, (-1, *phi2.shape.as_list()[-3:]))[:, :, :, -1:]
            xr2 = tf.clip_by_value((xr2 - self.ph_mean) / self.ph_std, -5.0, 5.0)
            print("xr2 shape ========== :",xr2.shape)


        # 基础网络用的是nature , 因为它的输入和我们的一样
        phi1 = self.natureHead(xr1)
        print("phi1 shape ========== :",phi1.shape)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            phi2 = self.natureHead(xr2)
            print("phi2 shape ========== :",phi2.shape)

            
        ac_one_hot = tf.one_hot(self.ph_ac, self.ac_space.n, axis=2)
        assert ac_one_hot.get_shape().ndims == 3
        assert ac_one_hot.get_shape().as_list() == [None, None, self.ac_space.n], ac_one_hot.get_shape().as_list()
        ac_one_hot = tf.reshape(ac_one_hot, (-1, self.ac_space.n))
        print("a one hot 9999999999999999999999999999:",ac_one_hot.shape)
            
        # inverse model: g(phi1,phi2) -> a_inv: [None, ac_space]
        g = tf.concat([phi1, phi2],1)
        print("g.shape 00000000000000000000000000000000000 :",g.shape)
        g = tf.nn.relu(self.linear(g, size, "g1", self.normalized_columns_initializer(0.01)))
        print("asample 000000000000000000000000000000:",asample.shape)
        aindex = tf.argmax(ac_one_hot, axis=1)  # aindex: [batch_size,]
        print("a index 99999999999999999999999999999999:",aindex.shape)
        # self.asample = asample = tf.placeholder(tf.float32, [None, ac_space]) 在原文中的大小
        logits = self.linear(g, ac_space.n, "glast", self.normalized_columns_initializer(0.01))
        print("logits 99999999999999999999999999999999:",logits.shape)
        self.invloss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                        logits=logits, labels=aindex), name="invloss")
        self.ainvprobs = tf.nn.softmax(logits, dim=-1)



        # forward model: f(phi1,asample) -> phi2
        # Note: no backprop to asample of policy: it is treated as fixed for predictor training
        f = tf.concat([phi1, ac_one_hot], 1)
        print("f.shape 000000000000000000000000000000 : ",f.shape) #(?, 519)
        f = tf.nn.relu(self.linear(f, size, "f1", self.normalized_columns_initializer(0.01)))
        f = self.linear(f, phi1.get_shape()[1].value, "flast", self.normalized_columns_initializer(0.01))
        self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
        # self.forwardloss = 0.5 * tf.reduce_mean(tf.sqrt(tf.abs(tf.subtract(f, phi2))), name='forwardloss')
        # self.forwardloss = cosineLoss(f, phi2, name='forwardloss')
        self.forwardloss = self.forwardloss * 288.0  # lenFeatures=288. Factored out to make hyperparams not depend on it.
        error = self.forwardloss # 这里的维度不一定对
        self.int_rew = error * constants['PREDICTION_BETA']


        self.aux_loss = constants['PREDICTION_LR_SCALE'] * (self.invloss * (1-constants['FORWARD_LOSS_WT']) +
                                                                    self.forwardloss * constants['FORWARD_LOSS_WT'])
        # self.ap_network= Predictor = StateActionPredictor(ob_space,ac_space)
        # self.predloss = constants['PREDICTION_LR_SCALE'] * (predictor.invloss * (1-constants['FORWARD_LOSS_WT']) +
        #                                                             predictor.forwardloss * constants['FORWARD_LOSS_WT'])

        # self.aux_loss = self.predloss
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)
    
    def define_self_prediction_rew(self, convfeat, rep_size, enlargement):
        #RND.
        # Random target network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xr = ph[:,1:]
                xr = tf.cast(xr, tf.float32)
                xr = tf.reshape(xr, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
                xr = tf.clip_by_value((xr - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xr = tf.nn.leaky_relu(conv(xr, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbr = [to2d(xr)]
                X_r = fc(rgbr[0], 'fc1r', nh=rep_size, init_scale=np.sqrt(2))

        # Predictor network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xrp = ph[:,1:]
                xrp = tf.cast(xrp, tf.float32)
                xrp = tf.reshape(xrp, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
                xrp = tf.clip_by_value((xrp - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xrp = tf.nn.leaky_relu(conv(xrp, 'c1rp_pred', nf=convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c2rp_pred', nf=convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c3rp_pred', nf=convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbrp = to2d(xrp)
                X_r_hat = tf.nn.relu(fc(rgbrp, 'fc1r_hat1_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(X_r_hat, 'fc1r_hat2_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = fc(X_r_hat, 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))

        self.feat_var = tf.reduce_mean(tf.nn.moments(X_r, axes=[0])[1])
        self.max_feat = tf.reduce_max(tf.abs(X_r))
        self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1, keep_dims=True)
        self.int_rew = tf.reshape(self.int_rew, (self.sy_nenvs, self.sy_nsteps - 1))

        noisy_targets = tf.stop_gradient(X_r)
        self.aux_loss = tf.reduce_mean(tf.square(noisy_targets - X_r_hat), -1)
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)

    def define_dynamics_prediction_rew(self, convfeat, rep_size, enlargement):
        #Dynamics based bonus.

        # Random target network.
        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xr = ph[:,1:]
                xr = tf.cast(xr, tf.float32)
                xr = tf.reshape(xr, (-1, *ph.shape.as_list()[-3:]))[:, :, :, -1:]
                xr = tf.clip_by_value((xr - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xr = tf.nn.leaky_relu(conv(xr, 'c1r', nf=convfeat * 1, rf=8, stride=4, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c2r', nf=convfeat * 2 * 1, rf=4, stride=2, init_scale=np.sqrt(2)))
                xr = tf.nn.leaky_relu(conv(xr, 'c3r', nf=convfeat * 2 * 1, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbr = [to2d(xr)]
                X_r = fc(rgbr[0], 'fc1r', nh=rep_size, init_scale=np.sqrt(2))

        # Predictor network.
        ac_one_hot = tf.one_hot(self.ph_ac, self.ac_space.n, axis=2)
        assert ac_one_hot.get_shape().ndims == 3
        assert ac_one_hot.get_shape().as_list() == [None, None, self.ac_space.n], ac_one_hot.get_shape().as_list()
        ac_one_hot = tf.reshape(ac_one_hot, (-1, self.ac_space.n))
        def cond(x):
            return tf.concat([x, ac_one_hot], 1)

        for ph in self.ph_ob.values():
            if len(ph.shape.as_list()) == 5:  # B,T,H,W,C
                logger.info("CnnTarget: using '%s' shape %s as image input" % (ph.name, str(ph.shape)))
                xrp = ph[:,:-1]
                xrp = tf.cast(xrp, tf.float32)
                xrp = tf.reshape(xrp, (-1, *ph.shape.as_list()[-3:]))
                # ph_mean, ph_std are 84x84x1, so we subtract the average of the last channel from all channels. Is this ok?
                xrp = tf.clip_by_value((xrp - self.ph_mean) / self.ph_std, -5.0, 5.0)

                xrp = tf.nn.leaky_relu(conv(xrp, 'c1rp_pred', nf=convfeat, rf=8, stride=4, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c2rp_pred', nf=convfeat * 2, rf=4, stride=2, init_scale=np.sqrt(2)))
                xrp = tf.nn.leaky_relu(conv(xrp, 'c3rp_pred', nf=convfeat * 2, rf=3, stride=1, init_scale=np.sqrt(2)))
                rgbrp = to2d(xrp)

                # X_r_hat = tf.nn.relu(fc(rgb[0], 'fc1r_hat1', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(cond(rgbrp), 'fc1r_hat1_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = tf.nn.relu(fc(cond(X_r_hat), 'fc1r_hat2_pred', nh=256 * enlargement, init_scale=np.sqrt(2)))
                X_r_hat = fc(cond(X_r_hat), 'fc1r_hat3_pred', nh=rep_size, init_scale=np.sqrt(2))

        self.feat_var = tf.reduce_mean(tf.nn.moments(X_r, axes=[0])[1])
        self.max_feat = tf.reduce_max(tf.abs(X_r))
        self.int_rew = tf.reduce_mean(tf.square(tf.stop_gradient(X_r) - X_r_hat), axis=-1, keep_dims=True)
        self.int_rew = tf.reshape(self.int_rew, (self.sy_nenvs, self.sy_nsteps - 1))

        noisy_targets = tf.stop_gradient(X_r)
        self.aux_loss = tf.reduce_mean(tf.square(noisy_targets - X_r_hat), -1)
        mask = tf.random_uniform(shape=tf.shape(self.aux_loss), minval=0., maxval=1., dtype=tf.float32)
        mask = tf.cast(mask < self.proportion_of_exp_used_for_predictor_update, tf.float32)
        self.aux_loss = tf.reduce_sum(mask * self.aux_loss) / tf.maximum(tf.reduce_sum(mask), 1.)

    def initial_state(self, n):
        return np.zeros((n, self.memsize), np.float32)

    def call(self, dict_obs, new, istate, update_obs_stats=False):
        for ob in dict_obs.values():
            if ob is not None:
                if update_obs_stats:
                    raise NotImplementedError
                    ob = ob.astype(np.float32)
                    ob = ob.reshape(-1, *self.ob_space.shape)
                    self.ob_rms.update(ob)
        # Note: if it fails here with ph vs observations inconsistency, check if you're loading agent from disk.
        # It will use whatever observation spaces saved to disk along with other ctor params.
        feed1 = { self.ph_ob[k]: dict_obs[k][:,None] for k in self.ph_ob_keys }
        feed2 = { self.ph_istate: istate, self.ph_new: new[:,None].astype(np.float32) }
        feed1.update({self.ph_mean: self.ob_rms.mean, self.ph_std: self.ob_rms.var ** 0.5})
        # for f in feed1:
        #     print(f)
        a, vpred_int,vpred_ext, nlp, newstate, ent = tf.get_default_session().run(
            [self.a_samp, self.vpred_int_rollout,self.vpred_ext_rollout, self.nlp_samp, self.snext_rollout, self.entropy_rollout],
            feed_dict={**feed1, **feed2})
        return a[:,0], vpred_int[:,0],vpred_ext[:,0], nlp[:,0], newstate, ent[:,0]

