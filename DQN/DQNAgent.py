import numpy as np
from DQN.DQNmodels import build_q_func
from DQN.DQN_utils import ObservationInput
from DQN.build_graph import build_train,build_act
import tensorflow as tf
import cloudpickle
import tempfile
import zipfile
from Common.tf_util import load_variables, save_variables
import os
from DQN.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from DQN.schedules import LinearSchedule

class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)



class DQNAgent():
    def __init__(self,args):
        print("agent, args",args)
        self.obs_size = list(args.obs_size)
        self.observation_space = args.observation_space
        self.n_actions = args.num_actions
        self.seed = args.seed
        self.rng = np.random.RandomState(self.seed)
        self.q_func = build_q_func(args.network)
        self.act, self.train, self.update_target, debug = build_train(
            make_obs_ph=self.make_obs_ph,
            q_func=self.q_func,
            num_actions=self.n_actions,
            optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr),
            gamma=args.gamma,
            grad_norm_clipping=10,
            param_noise=args.param_noise
        )# 在build——graph 中完成

        self.act_params = {
            'make_obs_ph': self.make_obs_ph,
            'q_func': self.q_func,
            'num_actions': self.n_actions,
        }

        self.act = ActWrapper(self.act, self.act_params)
        if args.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(args.buffer_size, alpha=args.prioritized_replay_alpha)
            if args.prioritized_replay_beta_iters is None:
                args.prioritized_replay_beta_iters = args.total_steps
            self.beta_schedule = LinearSchedule(args.prioritized_replay_beta_iters,
                                       initial_p=args.prioritized_replay_beta0,
                                       final_p=1.0)
        else:
            self.replay_buffer = ReplayBuffer(args.buffer_size)
            self.beta_schedule = None
        # Create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(schedule_timesteps=int(args.exploration_fraction * args.total_steps),
                                 initial_p=1.0,
                                 final_p=args.exploration_final_eps)


    def make_obs_ph(self,name):
        return ObservationInput(self.observation_space, name=name)

    def GetRandomAction(self,obs):
        action = self.rng.randint(0, self.n_actions)
        return action