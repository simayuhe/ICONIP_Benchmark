#runner for sparsity
# 记录共有多少个回合，每个回合有多少次奖励，要把环境的名字加上
import numpy as np

def run(env,agent,args):
    nenvs=args.num_env
    obses = env.reset()
    r_nenv= [[] for i in range(nenvs)]
    l_nenv= [[] for i in range(nenvs)]
    for t in range(args.total_steps):
        acts  = agent.GetRandomAction()
        obses_,rews,dones,infos = env.step(acts)
        dones=list(dones)
        obses= obses_
        if True in dones:
            for i,info in enumerate(infos):
                maybeinfo = info.get("r")
                if maybeinfo:
                    r_nenv[i].append(info["r"])
                    l_nenv[i].append(info["l"])
    np.save("./Results/Sparsity/"+args.env+"_rewlist.npy",r_nenv)
    np.save("./Results/Sparsity/"+args.env+"_lenlist.npy",l_nenv)
            
    return True