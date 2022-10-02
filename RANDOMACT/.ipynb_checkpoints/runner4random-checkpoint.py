from Common.showgif import save_frames_as_gif

def run(env,agent,args):
    nenvs=args.num_env
    obses = env.reset()
    frames = [[] for i in range(nenvs)]
    for t in range(args.total_steps):
        acts  = agent.GetRandomAction()
        obses_,rews,dones,infos = env.step(acts)
        print(infos)
        for i in range(nenvs):
            frames[i].append(obses[i,:,:,:])
        dones=list(dones)
        obses= obses_
        if True in dones:
            ith_env_done = dones.index(True)
            print("we done ",ith_env_done," at", t)
            save_frames_as_gif(frames[ith_env_done],filename="{}".format(args.env+str(ith_env_done)))
    return True