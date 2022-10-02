import numpy as np
from A2C.utils import discount_with_dones
from Common.runners import AbstractEnvRunner                

class Runner(AbstractEnvRunner):
    """
    We use this class to generate batches of experiences
                
    __init__:
    - Initialize the runner

    run():
    - Make a mini batch of experiences
    """
    def __init__(self, env, model, nsteps=5, gamma=0.99):
        super().__init__(env=env, model=model, nsteps=nsteps)
        self.gamma = gamma
#         self.batch_action_shape = [x if x is not None else -1 for x in model.train_model.action.shape.as_list()]
#         self.ob_dtype = model.train_model.X.dtype.as_numpy_dtype

    def run(self):
        # We initialize the lists that will contain the mb of experiences
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_raw_rewards = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for n in range(self.nsteps):
            actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs,raw_rewards, dones, infos = self.env.step(actions)
#             print(infos)
            for info in infos:
                maybeepinfo = info.get('r')#把所有有关r的拿出来
                if maybeepinfo: epinfos.append(info)#重新组合一下传回去
            rewards = np.sign(raw_rewards)
            self.states = states
            self.dones = dones
            if hasattr(self.model, 'sil'): 
                self.model.sil.step(self.obs, actions, raw_rewards, dones) 
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.obs = obs
            mb_rewards.append(rewards)
            mb_raw_rewards.append(raw_rewards)
        mb_dones.append(self.dones)

        # Batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_raw_rewards = np.asarray(mb_raw_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()


        if self.gamma > 0.0:
            # Discount/bootstrap off value fn
            last_values = self.model.value(self.obs, S=self.states, M=self.dones).tolist()
            for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
                rewards = rewards.tolist()
                dones = dones.tolist()
                if dones[-1] == 0:
                    rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
                else:
                    rewards = discount_with_dones(rewards, dones, self.gamma)

                mb_rewards[n] = rewards

        mb_rewards = mb_rewards.flatten()
        mb_raw_rewards = mb_raw_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_raw_rewards,epinfos