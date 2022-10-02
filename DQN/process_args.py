import numpy as np

def deepq_env_args(args):
    env_args = args
    env_args.noop_max = 5
    env_args.max_episode_steps = 5000
    env_args.episode_life = True
    env_args.scale = False
    env_args.clip_rewards = True
    env_args.frame_stack = True
    return env_args

def deepq_agent_args(env,args):
    agent_args = args
    agent_args.obs_size=env.observation_space.shape
    agent_args.observation_space = env.observation_space
    agent_args.num_actions=env.action_space.n 
    #agent = RandomAgent(agent_args)
    agent_args.network = "conv_only"
    agent_args.gamma = 0.99
    agent_args.param_noise = False
    agent_args.lr = 1e-4
    agent_args.prioritized_replay = True
    agent_args.buffer_size= 10000
    agent_args.prioritized_replay_alpha=0.6
    agent_args.prioritized_replay_beta_iters = None
    agent_args.prioritized_replay_beta0 = 0.4
    agent_args.prioritized_replay_eps=1e-6
    agent_args.exploration_fraction=0.1
    agent_args.exploration_final_eps=0.01
    return agent_args


def deepq_runner_args(env,args):
    runner_args = args
    runner_args.learning_starts = 10000
    runner_args.batch_size = 32
    runner_args.train_freq = 4
    runner_args.target_network_update_freq = 5000 #500
    runner_args.model_file = "deepq.pkl"
    # runner_args.load_path = "deepq.pkl"
    runner_args.checkpoint_freq = 5000
    return runner_args