import random
class RANDOMAgent():
    def __init__(self,args):
        self.num_actions = args.num_actions
        self.num_env = args.num_env
        
    def GetRandomAction(self):
        if self.num_env>1:
            acts = [random.randint(0,self.num_actions-1) for i in range(self.num_env)]
        else:
            acts = random.randint(0,self.num_actions-1) 

        return acts
        
        