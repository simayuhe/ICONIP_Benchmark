
def hashcount_env_args(args):
    env_args = args
    env_args.noop_max = 30
    env_args.max_episode_steps = 5000
    env_args.episode_life = True
    env_args.scale = False
    env_args.clip_rewards = False
    env_args.frame_stack = False
    env_args.num_env = 32# hash # rnd 4
    return env_args

def hashcount_agent_args(env,args):
    agent_args= args
    agent_args.policy="rnn"#"cnn"# 这个有可能不对,源代码参数名是policy
    agent_args.gamma = 0.99
    agent_args.gamma_ext = 0.999# 换成0.999
    agent_args.lam = 0.95
    agent_args.update_ob_stats_every_step=0
    agent_args.update_ob_stats_independently_per_gpu=0
    agent_args.update_ob_stats_from_random_agent=1
    agent_args.proportion_of_exp_used_for_predictor_update = 1.0
    agent_args.int_coeff= 1.0
    agent_args.ext_coeff=2.0
    agent_args.dynamics_bonus = 0
    agent_args.use_news= 0
    agent_args.nminibatches = 4
    agent_args.nepochs = 4
    agent_args.lr = 0.0001# 这里的learningrate比别人要小、
    agent_args.max_grad_norm = 0.0
    agent_args.testing = False
    agent_args.cliprange = 0.1# 应该是可调的，但是不知道调完之后的效果
    agent_args.nsteps = 128# hash #rnd 128
    agent_args.vf_coef = 1.0
    agent_args.ent_coef = 0.001
    return agent_args

def hashcount_runner_args(env,args):
    runner_args = args
    runner_args.learning_starts = 10000
    runner_args.checkpoint_freq = 2560
    return runner_args