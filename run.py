import sys
from Common.cmd_utils import common_arg_parser

def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    print(args,unknown_args)

  
    if args.alg =="RANDOMACT":
        from RANDOMACT.process_args import randomact_env_args
        # from EnvBuilder.env_builder import build_env
        from EnvBuilder.vec_env_builder import build_vec_env
        env_args = randomact_env_args(args)
        print("random env parser")
        env=build_vec_env(env_args)
        # env =  build_env(env_args)
        print("env",env)
        
        from RANDOMACT.process_args import randomact_agent_args
        from RANDOMACT.RANDOMAgent import RANDOMAgent
        agent_args = randomact_agent_args(env,args)
        print("agent _args",agent_args)
        agent = RANDOMAgent(agent_args)
        
        
        from RANDOMACT.process_args import randomact_runner_args
        from RANDOMACT.runner4random import run
        runner_args = randomact_runner_args(env,args)
        print("runner_args",runner_args)
        run_flag = run(env,agent,runner_args)
        print("run_flag",run_flag)

    if args.alg =="SPARSITY":
        from SPARSITY.process_args import sparsity_env_args
        from EnvBuilder.vec_env_builder import build_vec_env
        env_args = sparsity_env_args(args)
        print("random env parser")
        env=build_vec_env(env_args)
        print("env",env)
        
        from SPARSITY.process_args import sparsity_agent_args
        from RANDOMACT.RANDOMAgent import RANDOMAgent
        agent_args = sparsity_agent_args(env,args)
        agent = RANDOMAgent(agent_args)
        
        
        from SPARSITY.process_args import sparsity_runner_args
        from SPARSITY.runner4sparsity import run
        runner_args = sparsity_runner_args(env,args)
        print("runner_args",runner_args)
        run_flag = run(env,agent,runner_args)
        print("run_flag",run_flag)

    if args.alg == "A2C":

        from A2C.a2c_process_args import a2c_env_args
        #from A2C.vec_env_builder import build_vec_env
        from EnvBuilder.vec_env_builder import build_vec_env
        env_args = a2c_env_args(args)
        print("a2c env parser",env_args)
        env=build_vec_env(env_args)
        print("env",env)
        
        from A2C.a2c_process_args import a2c_agent_args
        from A2C.A2CAgent import A2CAgent
        agent_args = a2c_agent_args(env,args)
        agent = A2CAgent(env,agent_args) 
                  
        from A2C.a2c_process_args import a2c_runner_args
        from A2C.runner4a2c import run
        runner_args = a2c_runner_args(env,args)
        print("runner_args",runner_args)
        run_flag = run(env,agent,runner_args)
        print("run_flag",run_flag)
    

        


    if args.alg =="DQN":
        from DQN.process_args import deepq_env_args
        from EnvBuilder.env_builder import build_env
        env_args = deepq_env_args(args)
        print("env ",env_args)
        env=build_env(env_args)
        

        from DQN.process_args import deepq_agent_args
        from DQN.DQNAgent import DQNAgent
        agent_args=deepq_agent_args(env,args)
        print("agent",agent_args)
        agent = DQNAgent(agent_args)
        

        from DQN.process_args import deepq_runner_args
        from DQN.runner4deepq import run
        runner_args = deepq_runner_args(env,args)
        print(runner_args)
        run_flag = run(env,agent,runner_args)

        # tester_args = args
        # # tester_args.testing = False
        # tester_args.test_model_file = "./models/m.pkl"
        # tester_args.test_steps = 10000
        # test_flag = test(env,agent,tester_args)
        # print(tester_args)# 用来测试当前模型的

    if args.alg=="PPO2": 
        from PPO2.ppo2_process_args import ppo2_env_args
        from EnvBuilder.vec_env_builder import build_vec_env
        env_args = ppo2_env_args(args)
        print("ppo2 env parser",env_args)
        env=build_vec_env(env_args)
        print("env",env)
        
        from PPO2.ppo2_process_args import ppo2_agent_args
        from PPO2.PPO2Agent import PPO2Agent
        agent_args = ppo2_agent_args(env,args)
        agent = PPO2Agent(env,agent_args)
        print("agent args",agent_args)
        
        from PPO2.ppo2_process_args import ppo2_runner_args
        from PPO2.runner4ppo2 import run
        runner_args = ppo2_runner_args(env,args)
        print("runner_args",runner_args)
        run_flag = run(env,agent,runner_args)
        print("run_flag",run_flag)

    if args.alg =="RND":
        from RND.rnd_process_args import rnd_env_args
        from EnvBuilder.vec_env_builder import build_vec_env
        env_args = rnd_env_args(args)
        env=build_vec_env(env_args)
        
        from RND.rnd_process_args import rnd_agent_args
        agent_args = rnd_agent_args(env,args)
        from RND.RNDAgent import RNDAgent
        agent = RNDAgent(env,agent_args)
        
        from RND.rnd_process_args import rnd_runner_args
        runner_args = rnd_runner_args(env,args)
        from RND.runner4rnd import run
        run_flag = run(env,agent,runner_args)
        
    
        
    if args.alg =="ICM":
        from ICM.icm_process_args import icm_env_args
        from EnvBuilder.vec_env_builder import build_vec_env
        env_args = icm_env_args(args)
        env=build_vec_env(env_args)
        
        from ICM.icm_process_args import icm_agent_args
        agent_args = icm_agent_args(env,args)
        from ICM.ICMAgent import ICMAgent
        agent = ICMAgent(env,agent_args)
        
        from ICM.icm_process_args import icm_runner_args
        runner_args = icm_runner_args(env,args)
        from ICM.runner4icm import run
        run_flag = run(env,agent,runner_args)

    if args.alg =="EMP":
        from EMP.emp_process_args import emp_env_args
        from EnvBuilder.vec_env_builder import build_vec_env
        env_args = emp_env_args(args)
        env=build_vec_env(env_args)
        
        from EMP.emp_process_args import emp_agent_args
        agent_args = emp_agent_args(env,args)
        from EMP.EMPAgent import EMPAgent
        agent = EMPAgent(env,agent_args)
        
        from EMP.emp_process_args import emp_runner_args
        runner_args = emp_runner_args(env,args)
        from EMP.runner4emp import run
        run_flag = run(env,agent,runner_args)
        
    if args.alg =="SIL":
        from SIL.sil_process_args import sil_env_args
        from EnvBuilder.vec_env_builder import build_vec_env
        env_args = sil_env_args(args)
        env=build_vec_env(env_args)
        
        from SIL.sil_process_args import sil_agent_args
        agent_args = sil_agent_args(env,args)
        from SIL.SILAgent import SILAgent
        agent = SILAgent(env,agent_args)
        
        from SIL.sil_process_args import sil_runner_args
        runner_args = sil_runner_args(env,args)
        from SIL.runner4sil import run
        run_flag = run(env,agent,runner_args)
        
    if args.alg == "HASHCOUNT":
        from HASHCOUNT.hashcount_process_args import hashcount_env_args
        from EnvBuilder.vec_env_builder import build_vec_env
        env_args = hashcount_env_args(args)
        env=build_vec_env(env_args)
        
        from HASHCOUNT.hashcount_process_args import hashcount_agent_args
        agent_args = hashcount_agent_args(env,args)
        from HASHCOUNT.HASHCOUNTAgent import HASHCOUNTAgent
        agent = HASHCOUNTAgent(env,agent_args)
        
        from HASHCOUNT.hashcount_process_args import hashcount_runner_args
        runner_args = hashcount_runner_args(env,args)
        from HASHCOUNT.runner4hashcount import run
        run_flag = run(env,agent,runner_args)

    
        
if __name__ == '__main__':
    print("hello world")
    main(sys.argv)