

# import time
# import functools
# import tensorflow as tf

# from baselines import logger

from Common.misc_util import set_global_seeds
# from baselines.common import tf_util
# from baselines.common.policies import build_policy


# from baselines.a2c.utils import Scheduler, find_trainable_variables
from SIL.Runner import Runner
# from baselines.ppo2.ppo2 import safemean
from collections import deque

# from tensorflow import losses
import numpy as np
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def run(env,agent,args):
    set_global_seeds(args.seed)
    runner = Runner(env, agent, nsteps=args.nsteps, gamma=args.gamma)
    rewdeq= deque(maxlen=100)
    nenvs = args.num_env
    nbatch = nenvs*args.nsteps
    print("we will update",args.total_steps//nbatch+1)
    ep_reward2save = []
    ep_reward_times2save = []
    ep_len2save = []
    for update in range(1,args.total_steps//nbatch+1):
#         obs, states, rewards, masks, actions, values, epinfos = runner.run()
        obs, states, rewards, masks, actions, values, raw_rewards,epinfos = runner.run()
#         print(epinfos)
#         print("mei ci gengxin ",len(obs))
        rewdeq.extend(epinfos)
        policy_loss, value_loss, policy_entropy, v_avg = agent.train(obs, states, rewards, masks, actions, values)
        sil_loss, sil_adv, sil_samples, sil_nlogp = agent.sil_train()#sil
        if update%args.checkpoint_freq ==0:
            meanrewards =safemean([epinfo['r'] for epinfo in rewdeq])
            meanrewards_positive = safemean([epinfo['r_p'] for epinfo in rewdeq])
            meanep_len = safemean([epinfo['l'] for epinfo in rewdeq])
            print("nupdates", update,"eprewmean", meanrewards)
            print("nupdates", update,"eprewmean_p",meanrewards_positive)
            print("nupdates", update,"ep_len", meanep_len)
            ep_reward2save.append(meanrewards)
            ep_reward_times2save.append(meanrewards_positive)
            ep_len2save.append(meanep_len)
            np.save(args.resultspath+args.env+"_SIL_"+"rew.npy",ep_reward2save)
            np.save(args.resultspath+args.env+"_SIL_"+"rew_times.npy",ep_reward_times2save)
            np.save(args.resultspath+args.env+"_SIL_"+"ep_len.npy",ep_len2save)
            