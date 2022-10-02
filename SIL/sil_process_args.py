import numpy as np

def sil_env_args(args):
    env_args = args
    env_args.noop_max = 30
    env_args.max_episode_steps = 5000
    env_args.episode_life = True
    env_args.scale = False
    env_args.clip_rewards = False
    env_args.frame_stack =False
    env_args.num_env = 32 # 这个数也不是越多越好的
    return env_args

def sil_agent_args(env,args):
    agent_args = args
    agent_args.nsteps = 5
    agent_args.num_actions=env.action_space.n 
    agent_args.ent_coef = 0.01
    agent_args.vf_coef = 0.5
    agent_args.max_grad_norm = 0.5 
    agent_args.lr = 7e-4
    agent_args.alpha = 0.99
    agent_args.epsilon = 1e-5
    agent_args.lrschedule = "linear"
    agent_args.network = "cnn"
    agent_args.sil_update = 4
    agent_args.sil_beta = 0.0
    return agent_args

def sil_runner_args(env,args):
    runner_args = args
    runner_args.gamma = 0.99
    runner_args.learning_starts = 10000
    runner_args.checkpoint_freq = 64
    return runner_args