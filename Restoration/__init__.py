import torch
import torch.nn as nn
from Restoration.degrationNet import UpdateZ,UpdateP,UpdateQ,UNET,PriorMixer,MultiScaleChannelTransformer
from utils import init_net
import itertools
import time

class Inference():
    def __init__(self,estimater_path,ref_dataset,T=6,mu_0=0.1,rou_0=0.1,increase=0.05):

        gpus = [0]
        self.estimater = torch.nn.DataParallel(init_net(UNET(in_channels=3,out_channels=6,base_dim=64)).eval(), device_ids=gpus, output_device=gpus[0])
        self.estimater.load_state_dict(torch.load(estimater_path))
        self.update_z = torch.nn.DataParallel(UpdateZ().train(), device_ids=gpus, output_device=gpus[0])
        self.update_p = torch.nn.DataParallel(UpdateP().train(), device_ids=gpus, output_device=gpus[0])
        self.update_q = torch.nn.DataParallel(UpdateQ().train(), device_ids=gpus, output_device=gpus[0])
        self.update_x = torch.nn.DataParallel(init_net(UNET(3,3,64)).train(), device_ids=gpus, output_device=gpus[0])
        self.update_ab = torch.nn.DataParallel(init_net(UNET(6,6,64)).train(), device_ids=gpus, output_device=gpus[0])
        self.prior_mixer = torch.nn.DataParallel(init_net(PriorMixer(base_dim=32)).train(), device_ids=gpus, output_device=gpus[0])

        self.T = T
        self.mu_0 = mu_0
        self.rou_0 = rou_0
        self.increase = increase
        self.ref_dataset = ref_dataset


    def __call__(self,y,labels):
        st = time.time()
        xs = []
        As = []
        Bs = []
        with torch.no_grad():
            x_0,Arec,B = self.estimater(y)
        Arec[Arec==0.0]+=1e-5
        x=x_0
        A = 1/Arec
        P = A
        Q = B
        mu = self.mu_0
        rou = self.rou_0
        for t in range(self.T):
            if t>0:
                imgB_ref, imgD_ref = self.ref_dataset.get(labels)
                imgB_ref = imgB_ref.cuda()
                imgD_ref = imgD_ref.cuda()
                P_ = self.update_p(Q, A, imgB_ref, imgD_ref, rou)
                Q = self.update_q(P, B, imgB_ref, imgD_ref, rou)
                P = P_
                A, B = self.prior_mixer(torch.cat([A, B],dim=1), self.update_ab(torch.cat([P, Q], dim=1)))
                rou = rou + self.increase
            As.append(A)
            Bs.append(B)
            z = self.update_z(A,B,x,y,mu)
            x = self.update_x(z)
            xs.append(x)
            mu = mu + t * self.increase
        print(time.time()-st)
        return xs,x_0

    def train(self):
        self.update_z.train()
        self.update_p.train()
        self.update_q.train()
        self.update_ab.train()
        self.update_x.train()
        self.prior_mixer.train()
    def eval(self):
        self.update_z.eval()
        self.update_p.eval()
        self.update_q.eval()
        self.update_ab.eval()
        self.update_x.eval()
        self.prior_mixer.eval()
    def load_state_dict(self,dic):
        self.update_x.load_state_dict(dic['update_x'])
        self.update_ab.load_state_dict(dic['update_ab'])
        self.prior_mixer.load_state_dict(dic['prior_mixer'])








