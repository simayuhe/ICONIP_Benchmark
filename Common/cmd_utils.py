def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--env_type', help='type of environment, used when the environment type cannot be automatically determined', type=str)
    parser.add_argument('--seed', help='RNG seed', type=int, default=None)
    parser.add_argument('--total_steps', type=int, default=10000000,
                       help='Number of steps to interaction with env')
    parser.add_argument("--alg",help="algorithm",type=str,default="DQN" )
    parser.add_argument("--usemodel",help="use an existing model",type=bool,default=False )
    parser.add_argument("--savepath",help="algorithm",type=str,default="./Models/")
    parser.add_argument("--resultspath",help="algorithm",type=str,default="./Results/")
    return parser

