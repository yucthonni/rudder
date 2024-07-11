import numpy as np
import torch

class ReplayBuffer:
    def __init__(self,args):
        super(ReplayBuffer,self).__init__()
        self.args = args
        self.s = np.zeros((args.max_steps, args.state_dim))
        self.a = np.zeros((args.max_steps, args.action_dim))
        self.a_logprob = np.zeros((args.max_steps, args.action_dim))
        self.r = np.zeros((args.max_steps, 1))
        self.s_ = np.zeros((args.max_steps, args.state_dim))
        self.dw = np.zeros((args.max_steps, 1))
        self.done = np.zeros((args.max_steps, 1))
        self.cnt = 0
        self.dvc = torch.device("cpu")
        if args.cuda:
            self.dvc = torch.device("cuda")
        self.max_steps = args.max_steps
        
    def push(self, state, action, logprob, reward, state_, dw, done):
        self.s[self.cnt%self.max_steps] = state
        self.a[self.cnt%self.max_steps] = action
        self.a_logprob[self.cnt%self.max_steps] = logprob
        self.r[self.cnt%self.max_steps] = reward
        self.s_[self.cnt%self.max_steps] = state_
        self.dw[self.cnt%self.max_steps] = dw
        self.done[self.cnt%self.max_steps] = done
        self.cnt += 1
        
    def toTensor(self):
        s = torch.tensor(self.s, dtype=torch.float).to(self.dvc)
        a = torch.tensor(self.a, dtype=torch.float).to(self.dvc)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float).to(self.dvc)
        r = torch.tensor(self.r, dtype=torch.float).to(self.dvc)
        s_ = torch.tensor(self.s_, dtype=torch.float).to(self.dvc)
        dw = torch.tensor(self.dw, dtype=torch.float).to(self.dvc)
        done = torch.tensor(self.done, dtype=torch.float).to(self.dvc)

        return s, a, a_logprob, r, s_, dw, done