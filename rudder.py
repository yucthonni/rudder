import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import value_pred
from torch.utils.data import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

class Rudder:
    def __init__(self,args):
        self.args = args
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.dvc = torch.device('cpu')
        if args.cuda:
            self.dvc = torch.device('cuda')
        
        self.predicter = value_pred(args).to(self.dvc)
        
        self.optimizer = optim.Adam(self.predicter.parameters(),
                                    lr=args.lr_lstm,
                                    eps=1e-5)
        self.criteria = nn.MSELoss()
        self.writer = SummaryWriter("")
        self.steps = 0
        
    def initWriter(self,logdir):
        self.writer = SummaryWriter(logdir)
        
    def update(self, buffer):
        s, a, a_logprob, r, s_, dw, done = buffer.toTensor()
        
        with torch.no_grad():
            dr = 0
            ret = []
            for i in reversed(range(len(s))):
                if done[i]:
                    dr = 0
                dr = r[i] + self.gamma * dr
                ret.insert(0, dr)
                
            ret = torch.stack(ret).to(self.dvc)
            
        for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)),self.mini_batch_size,False):
            pred = self.predicter(s[index],a[index])
            loss = self.criteria(pred,ret[index])
            
            self.writer.add_scalar("loss", loss, self.steps)
            self.steps += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        