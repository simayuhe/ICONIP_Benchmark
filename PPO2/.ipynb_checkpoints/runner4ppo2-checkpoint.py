# from Common.misc_util import set_global_seeds
import os
import time
import numpy as np
import os.path as osp
# from baselines import logger
from collections import deque
from Common.misc_util import set_global_seeds
# from baselines.common.policies import build_policy
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from PPO2.Runner import Runner

def constfn(val):
    def f(_):
        return val
    return f

def run(env,agent,args):
    seed = args.seed
    set_global_seeds(seed)
    lr = args.lr
    cliprange = args.cliprange
    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = args.total_steps
    nenvs = args.num_env
    nsteps= args.nsteps
    nminibatches = args.nminibatches
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    gamma = args.gamma
    eval_env = None
    init_fn = None
    update_fn = None
    noptepochs = args.noptepochs
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)
    load_path = args.load_path
    lam = args.lam
    if load_path is not None:
        model.load(load_path)
    runner = Runner(env=env, model=agent, nsteps=nsteps, gamma=gamma, lam=lam)
    if eval_env is not None:
        eval_runner = Runner(env = eval_env, model = agent, nsteps = nsteps, gamma = gamma, lam= lam)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()

    # Start total timer
    tfirststart = time.perf_counter()

    nupdates = total_timesteps//nbatch
    ep_reward2save = []
    ep_reward_times2save = []
    ep_len2save = []
    for update in range(1, nupdates+1):
#         print(update)
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        if update % args.checkpoint_freq == 0 and is_mpi_root: print('Stepping environment...')

        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run() #pylint: disable=E0632

        if update % args.checkpoint_freq == 0 and is_mpi_root: print('Done.')

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None: # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(agent.train(lrnow, cliprangenow, *slices))
        else: # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(agent.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))

        if update_fn is not None:
            update_fn(update)

        if update % args.checkpoint_freq == 0 or update == 1:
            meanrewards =safemean([epinfo['r'] for epinfo in epinfobuf])
            meanrewards_positive = safemean([epinfo['r_p'] for epinfo in epinfobuf])
            meanep_len = safemean([epinfo['l'] for epinfo in epinfobuf])
            print("nupdates", update,"eprewmean", meanrewards)
            print("nupdates", update,"eprewmean_p",meanrewards_positive)
            print("nupdates", update,"ep_len", meanep_len)
            ep_reward2save.append(meanrewards)
            ep_reward_times2save.append(meanrewards_positive)
            ep_len2save.append(meanep_len)
            np.save(args.resultspath+args.env+"_PPO_"+"rew.npy",ep_reward2save)
            np.save(args.resultspath+args.env+"_PPO_"+"rew_times.npy",ep_reward_times2save)
            np.save(args.resultspath+args.env+"_PPO_"+"ep_len.npy",ep_len2save)
            
            
    return True
    
# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
#             ev = explained_variance(values, returns)
#             logger.logkv("misc/serial_timesteps", update*nsteps)
#             logger.logkv("misc/nupdates", update)
#             logger.logkv("misc/total_timesteps", update*nbatch)
#             logger.logkv("fps", fps)
#             logger.logkv("misc/explained_variance", float(ev))
#             logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
#             logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
#             if eval_env is not None:
#                 logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]) )
#                 logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]) )
#             logger.logkv('misc/time_elapsed', tnow - tfirststart)
#             for (lossval, lossname) in zip(lossvals, model.loss_names):
#                 logger.logkv('loss/' + lossname, lossval)

#             logger.dumpkvs()
#         if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and is_mpi_root:
#             checkdir = osp.join(logger.get_dir(), 'checkpoints')
#             os.makedirs(checkdir, exist_ok=True)
#             savepath = osp.join(checkdir, '%.5i'%update)
#             print('Saving to', savepath)
#             model.save(savepath)