import numpy as np

def sparsity_env_args(args):
    env_args = args
    env_args.noop_max = 30
    env_args.max_episode_steps = 5000
    env_args.episode_life = True
    env_args.scale = False
    env_args.clip_rewards = True
    env_args.frame_stack = False
    env_args.num_env = 4
    return env_args

def sparsity_agent_args(env,args):
    agent_args = args
#     agent_args.obs_size=env[0].observation_space.shape
#     agent_args.observation_space = env[0].observation_space
    agent_args.num_actions=env.action_space.n 
    return agent_args

def sparsity_runner_args(env,args):
    runner_args = args
    runner_args.learning_starts = 10000
    runner_args.checkpoint_freq = 5000
    return runner_args