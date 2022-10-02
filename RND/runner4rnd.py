import RND.mpi_util as mpi_util
import RND.tf_util as tf_util
import os 
from Common.misc_util import set_global_seeds
from mpi4py import MPI
import numpy as np
from collections import deque
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def run(env,agent,args):
    env.score_multiple = 1# 干啥用的？
    mpi_util.setup_mpi_gpus()
    
    seed = 10000 * args.seed + MPI.COMM_WORLD.Get_rank()
    set_global_seeds(seed)
    agent.start_interaction([env])
    if args.update_ob_stats_from_random_agent:
        agent.collect_random_statistics(num_timesteps=128*50)
    counter = 0
#     npzeri=np.zeros(1000000)
    epinfolist = deque(maxlen=100)
    ep_reward2save = []
    ep_reward_times2save = []
    ep_len2save = []
    while True:
#         print("counter",counter)
        counter+=1
        epinfos,stepcount = agent.step()
        # print("ep,stepcount",epinfos,stepcount)
        epinfolist.extend(epinfos)
        if counter%args.checkpoint_freq==0:
            meanrewards =safemean([epinfo['r'] for epinfo in epinfolist])
            meanrewards_positive = safemean([epinfo['r_p'] for epinfo in epinfolist])
            meanep_len = safemean([epinfo['l'] for epinfo in epinfolist])
            print("nupdates", counter,"eprewmean", meanrewards)
            print("nupdates", counter,"eprewmean_p",meanrewards_positive)
            print("nupdates", counter,"ep_len", meanep_len)
            ep_reward2save.append(meanrewards)
            ep_reward_times2save.append(meanrewards_positive)
            ep_len2save.append(meanep_len)
            np.save(args.resultspath+args.env+"_RND_"+"rew.npy",ep_reward2save)
            np.save(args.resultspath+args.env+"_RND_"+"rew_times.npy",ep_reward_times2save)
            np.save(args.resultspath+args.env+"_RND_"+"ep_len.npy",ep_len2save)
#         if info['update']:
#             print(counter)
#             counter += 1
#             if counter>2:
#                 npzeri[counter]=int(info['update']['eprew'])
#             np.save('a.npy',npzeri)
        if counter > args.total_steps:
            break

    agent.stop_interaction()
    return True