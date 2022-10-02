import gym
# import sacred
# from sacred.observers import FileStorageObserver
import matplotlib.pyplot as plt
from matplotlib import animation


def plotframes(frames,filename):
    for i in range(len(frames)):
        plt.imshow(frames[i])
        plt.savefig("./tmp/"+filename+str(i)+".png")
        plt.close()

def save_frames_as_gif(frames,filename='gym_animation'):  
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    filename_gif = "{}.gif".format(filename)
    path = "{}.gif".format(filename)
    anim.save(path, writer='imagemagick', fps=60)
    
def test_env(env_name: str, rgb):
    env: gym.Env = gym.make(env_name)
    s = env.reset()
    frames = []
    for step in range(10000):
        if rgb:
            frames.append(env.render(mode="rgb_array"))
        else:
            env.render()
        a = env.action_space.sample()
        ns, r, d, i = env.step(a)
        if d:
            break
        s = ns
    env.close()
    if rgb:
        save_frames_as_gif(frames, filename="{}".format(env_name))
        
if __name__ == '__main__':
    print("be")
    test_env("Pong-v0",True)