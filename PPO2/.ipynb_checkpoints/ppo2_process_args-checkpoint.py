import numpy as np


def ppo2_env_args(args):
    env_args = args
    env_args.noop_max = 30
    env_args.max_episode_steps = 5000
    env_args.episode_life = True
    env_args.scale = False
    env_args.clip_rewards = True
    env_args.frame_stack = False
    env_args.num_env = 4
    return env_args    

def ppo2_agent_args(env,args):
    agent_args = args
    agent_args.nsteps = 128#2048
    agent_args.nminibatches = 4
    agent_args.network = "cnn"
    agent_args.num_actions=env.action_space.n
    agent_args.ent_coef = 0.01
    agent_args.vf_coef = 0.5
    agent_args.max_grad_norm = 0.5
    agent_args.gamma = 0.99
    agent_args.lam = 0.95
    agent_args.lr = lambda f : f * 2.5e-4
    agent_args.cliprange=0.1
    agent_args.noptepochs = 4
    return agent_args

def ppo2_runner_args(env,args):
    runner_args = args
    runner_args.load_path = None
    runner_args.learning_starts = 10000
    runner_args.checkpoint_freq = 100
    return runner_args