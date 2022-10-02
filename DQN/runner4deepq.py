
from json import load
from Common.tf_util import get_session
from Common.misc_util import set_global_seeds
import Common.tf_util as U
import numpy as np
from Common.tf_util import load_variables, save_variables
import tensorflow as tf

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def run(env,agent,args):
    sess = get_session()
    config = tf.compat.v1.ConfigProto()
    # # rtx 3090
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # config.gpu_options.allow_growth = True
    # config.gpu_options.polling_inactive_delay_msecs = 10
    # sess = tf.compat.v1.Session(config=config)

    set_global_seeds(args.seed)
    U.initialize()
    agent.update_target()
    obs = env.reset()
    episode_rewards = [0.0]
    eprew_timesmean_list = []
    td = args.model_file
    # if args.usemodel:
    #     load_variables(args.savepath+args.model_file)
    # if tf.train.latest_checkpoint(td) is not None:
    #     load_variables(args.model_file)
    #     print("load model from :", args.model_file)
    #     model_saved = True
    # elif args.load_path is not None:
    #     load_variables(args.load_path)
    #     print("load model from path",args.load_path)


    # input("have loaded successful!")
    for t in range(args.total_steps):
        kwargs={}
        if not args.param_noise:
            update_eps = agent.exploration.value(t)
            update_param_noise_threshold = 0.
        else:
            update_eps = 0.
            # Compute the threshold such that the KL divergence between perturbed and non-perturbed
            # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
            # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
            # for detailed explanation.
            update_param_noise_threshold = -np.log(1. - agent.exploration.value(t) + agent.exploration.value(t) / float(env.action_space.n))
            # kwargs['reset'] = reset
            kwargs['update_param_noise_threshold'] = update_param_noise_threshold
            kwargs['update_param_noise_scale'] = True
        act = agent.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
        new_obs, rew, done, info= env.step(act)
        episode_rewards[-1] += rew
        # print(t,act, rew,info)
        agent.replay_buffer.add(obs, act, rew, new_obs, float(done))
        obs = new_obs
        if done:
            obs = env.reset()
            print("done!current t:",t," episode reward: ",episode_rewards[-1])
            episode_rewards.append(0.0)
            # break
        if t>args.learning_starts and t % args.train_freq == 0:
            
            if args.prioritized_replay:
                experience = agent.replay_buffer.sample(args.batch_size, beta=agent.beta_schedule.value(t))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = agent.replay_buffer.sample(args.batch_size)
                weights, batch_idxes = np.ones_like(rewards), None
            td_errors = agent.train(obses_t, actions, rewards, obses_tp1, dones, weights)
            # print(td_errors)
            if args.prioritized_replay:
                new_priorities = np.abs(td_errors) + args.prioritized_replay_eps
                agent.replay_buffer.update_priorities(batch_idxes, new_priorities)

        if t > args.learning_starts and t % args.target_network_update_freq == 0:
            # Update target network periodically.
            print(" Update target network periodically")
            agent.update_target()
        if t > args.learning_starts and t % args.checkpoint_freq == 0:
            print("save model")
            save_variables(args.savepath+args.model_file)
            eprew_timesmean_list.append(safemean(episode_rewards))
            np.save("./Results/"+str(args.env)+str(args.alg)+str(args.checkpoint_freq)+"rew_times_list.npy",eprew_timesmean_list)

    return True