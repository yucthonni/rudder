import numpy as np
from arguments import get_args
from replaybuffer import ReplayBuffer
from rudder import Rudder
from tqdm import tqdm
import torch,os,datetime,gym

args = get_args()
env = gym.make("Pendulum-v0")
replaybuffer = ReplayBuffer(args)
agent = Rudder(args)
checkpoint = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
agent.initWriter(args.log_dir)
cri = torch.nn.MSELoss()

s = env.reset()
d = False
for i in range(args.max_steps):
    a = env.action_space.sample()
    s_, r, d, _ = env.step(a)
    
    if d & args.max_steps!=i+1:
        dw = d
    else:
        dw = False
        
    replaybuffer.push(s,a,None,r,s_,dw,d)
    s=s_
    
for _ in tqdm(range(int(1e3)),leave=False):
    agent.update(replaybuffer)
    
s = env .reset()
d = False
acc = []
s_batch = []
a_batch = []
r_batch = []
for i in range(100):
    a = env.action_space.sample()
    s_, r, d, _ = env.step(a)
    
    s_batch.append(s)
    a_batch.append(a)
    r_batch.append(r)
    
s_batch = torch.tensor(s_batch,dtype=torch.float).cuda()
a_batch = torch.tensor(a_batch,dtype=torch.float).cuda()
pred = agent.predicter(s_batch,a_batch)
for i in range(len(s_batch)-1):
    acc.append(cri(pred[i]-pred[i+1],torch.tensor(r_batch[i],dtype=torch.float32).cuda()))
   
acc = torch.stack(acc)    
print('acc:',acc.mean(),acc.std())

torch.save(agent.predicter.state_dict(),'./model.pth')
    
# s, a, _, _, _, _, _ = replaybuffer.toTensor()
# print(agent.predicter(s,a))
    
        
