import torch
import torch.nn as nn
from Restoration.degrationNet import UpdateZ,UpdateP,UpdateQ,DPT
from Unet import UNET
from utils import init_net
import itertools

class Inference():
    def __init__(self,estimater_path,ref_dataset,T=6,gamma=0.1,alpha=0.5,beta=0.5,increase=0.05):
        self.estimater = init_net(UNET(in_c=3,out_c=6,base_dim=16)).eval()
        self.estimater.load_state_dict(torch.load(estimater_path))
        self.update_z = init_net(UpdateZ()).train()
        self.update_p = init_net(UpdateP()).train()
        self.update_q = init_net(UpdateQ()).train()
        self.update_x = init_net(UNET(in_c=3,out_c=3,base_dim=32)).train()
        self.update_td = init_net(UNET(in_c=6,out_c=6,base_dim=16)).train()
        self.dpt = init_net(DPT(base_dim=16)).train()

        self.T = T
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.increase = increase
        self.ref_dataset = ref_dataset


    def __call__(self,y,labels):
        xs = []
        td_s = []
        with torch.no_grad():
            I,T,D = self.estimater(y)
        x=y
        xs.append(I)
        gamma = self.gamma
        alpha = self.alpha
        beta = self.beta
        imgB_ref, imgO_ref = self.ref_dataset.get(labels)
        imgB_ref = imgB_ref.cuda()
        imgO_ref = imgO_ref.cuda()

        # with torch.no_grad():
        #     I_,T_,D_ = self.estimater(imgO_ref)
        P = T
        Q = D
        for t in range(self.T):
            if t>0:
                P_new = self.update_p(Q,T,imgB_ref, imgO_ref,alpha)
                Q = self.update_q(P,D,imgB_ref, imgO_ref,beta)
                P = P_new
                TD_ = self.update_td(torch.cat([P, Q], dim=1))
                td_s.append(TD_)
                T, D = self.dpt(TD_,x,torch.cat([T, D],dim=1))
                alpha+=self.increase
                beta+=self.increase
            z = self.update_z(T,D,x,y,gamma)
            x = self.update_x(z)
            xs.append(x)
            gamma +=t * self.increase
        return xs,td_s,imgB_ref, imgO_ref

    def train(self):
        self.update_z.train()
        self.update_p.train()
        self.update_q.train()
        self.update_td.train()
        self.update_x.train()
        self.dpt.train()
    def eval(self):
        self.update_z.eval()
        self.update_p.eval()
        self.update_q.eval()
        self.update_td.eval()
        self.update_x.eval()
        self.dpt.eval()
    def load_state_dict(self,dic):
        self.update_x.load_state_dict(dic['update_x'])
        self.update_td.load_state_dict(dic['update_td'])
        self.dpt.load_state_dict(dic['dpt'])





