import torch
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class TrainDataSet(Dataset):
    def __init__(self,rain_paths=(''),
                 haze_paths=('')
                 ,dark_paths=('',''),img_size=256,size = 1200):
        super(TrainDataSet,self).__init__()
        self.img_size = img_size
        self.rain_paths = rain_paths
        self.haze_paths = haze_paths
        self.dark_paths = dark_paths


        rain800_imgs = [os.path.join(self.rain_paths[0], 'rain',img_i) for img_i in os.listdir(os.path.join(self.rain_paths[0], 'rain'))]
        rain800_bgs = [os.path.join(self.rain_paths[0], 'norain',img_i) for img_i in os.listdir(os.path.join(self.rain_paths[0], 'rain'))]
        rain12000_imgs = [os.path.join(self.rain_paths[1], 'rain',img_i) for img_i in os.listdir(os.path.join(self.rain_paths[1],'rain'))]
        rain12000_bgs = [os.path.join(self.rain_paths[1], 'norain',img_i) for img_i in os.listdir(os.path.join(self.rain_paths[1],'rain'))]
        self.rain_imgs = rain800_imgs+rain12000_imgs
        self.rain_bgs = rain800_bgs+rain12000_bgs

        its_imgs = [os.path.join(self.haze_paths[0],'hazy',img_i) for img_i in os.listdir(os.path.join(self.haze_paths[0],'hazy'))]
        its_bgs = [os.path.join(self.haze_paths[0],'clear',img_i.split('_')[0]+'.png') for img_i in os.listdir(os.path.join(self.haze_paths[0],'hazy'))]
        ots_imgs = [os.path.join(self.haze_paths[1],'part1',img_i) for img_i in os.listdir(os.path.join(self.haze_paths[1],'part1'))]
        ots_bgs = [os.path.join(self.haze_paths[1],'clear',img_i.split('_')[0]+'.jpg') for img_i in os.listdir(os.path.join(self.haze_paths[1],'part1'))]
        self.haze_imgs = its_imgs+ots_imgs
        self.haze_bgs = its_bgs+ots_bgs

        LOL_imgs = [os.path.join(self.dark_paths[0],'low',img_i) for img_i in os.listdir(os.path.join(self.dark_paths[0],'low'))]
        LOL_bgs = [os.path.join(self.dark_paths[0],'high',img_i) for img_i in os.listdir(os.path.join(self.dark_paths[0],'low'))]
        LSRW_imgs = [os.path.join(self.dark_paths[1], 'low', img_i) for img_i in os.listdir(os.path.join(self.dark_paths[1], 'low'))]
        LSRW_bgs = [os.path.join(self.dark_paths[1], 'high', img_i) for img_i in os.listdir(os.path.join(self.dark_paths[1], 'low'))]
        self.dark_imgs = LOL_imgs+LSRW_imgs
        self.dark_bgs = LOL_bgs+LSRW_bgs

        self.paired_transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.n_rain = len(self.rain_imgs)
        self.n_haze = len(self.haze_imgs)
        self.n_dark = len(self.dark_imgs)

        self.size = 1200
    def __getitem__(self, index):
        if index<self.size//3: # rain
            label = 0
            i = np.random.randint(0,self.n_rain)
            imgD = Image.open(self.rain_imgs[i]).convert('RGB')
            imgB = Image.open(self.rain_bgs[i]).convert('RGB')
        elif index<self.size//3*2: # haze
            label = 1
            i = np.random.randint(0,self.n_haze)
            imgD = Image.open(self.haze_imgs[i]).convert('RGB')
            imgB = Image.open(self.haze_bgs[i]).convert('RGB')
        else: # dark
            label = 2
            i = np.random.randint(0, self.n_dark)
            imgD = Image.open(self.dark_imgs[i]).convert('RGB')
            imgB = Image.open(self.dark_bgs[i]).convert('RGB')


        if np.random.random() < 0.5:
            w, h = imgD.size
            if w>self.img_size and h>self.img_size:
                dw = w-self.img_size
                dh = h-self.img_size
                ws = np.random.randint(dw+1)
                hs = np.random.randint(dh+1)
                imgD = imgD.crop((ws, hs, ws+self.img_size, hs+self.img_size))
                imgB = imgB.crop((ws, hs, ws + self.img_size, hs + self.img_size))
        if np.random.random() < 0.5:
            imgD= imgD.transpose(Image.FLIP_LEFT_RIGHT)
            imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)

        imgD = self.paired_transform(imgD)
        imgB = self.paired_transform(imgB)


        return [imgB,imgD,label]
    def __len__(self):
        return self.size



class RefDataset():
    def __init__(self,rain_paths=('',''),
                 haze_paths=('','')
                 ,dark_paths=('',''),img_size=256):
        self.img_size = img_size
        self.rain_paths = rain_paths
        self.haze_paths = haze_paths
        self.dark_paths = dark_paths


        rain800_imgs = [os.path.join(self.rain_paths[0], 'rain',img_i) for img_i in os.listdir(os.path.join(self.rain_paths[0], 'rain'))]
        rain800_bgs = [os.path.join(self.rain_paths[0], 'norain',img_i) for img_i in os.listdir(os.path.join(self.rain_paths[0], 'rain'))]
        rain12000_imgs = [os.path.join(self.rain_paths[1], 'rain',img_i) for img_i in os.listdir(os.path.join(self.rain_paths[1],'rain'))]
        rain12000_bgs = [os.path.join(self.rain_paths[1], 'norain',img_i) for img_i in os.listdir(os.path.join(self.rain_paths[1],'rain'))]
        self.rain_imgs = rain800_imgs+rain12000_imgs
        self.rain_bgs = rain800_bgs+rain12000_bgs

        its_imgs = [os.path.join(self.haze_paths[0],'hazy',img_i) for img_i in os.listdir(os.path.join(self.haze_paths[0],'hazy'))]
        its_bgs = [os.path.join(self.haze_paths[0],'clear',img_i.split('_')[0]+'.png') for img_i in os.listdir(os.path.join(self.haze_paths[0],'hazy'))]
        ots_imgs = [os.path.join(self.haze_paths[1],'part1',img_i) for img_i in os.listdir(os.path.join(self.haze_paths[1],'part1'))]
        ots_bgs = [os.path.join(self.haze_paths[1],'clear',img_i.split('_')[0]+'.jpg') for img_i in os.listdir(os.path.join(self.haze_paths[1],'part1'))]
        self.haze_imgs = its_imgs+ots_imgs
        self.haze_bgs = its_bgs+ots_bgs

        LOL_imgs = [os.path.join(self.dark_paths[0],'low',img_i) for img_i in os.listdir(os.path.join(self.dark_paths[0],'low'))]
        LOL_bgs = [os.path.join(self.dark_paths[0],'high',img_i) for img_i in os.listdir(os.path.join(self.dark_paths[0],'low'))]
        LSRW_imgs = [os.path.join(self.dark_paths[1], 'low', img_i) for img_i in os.listdir(os.path.join(self.dark_paths[1], 'low'))]
        LSRW_bgs = [os.path.join(self.dark_paths[1], 'high', img_i) for img_i in os.listdir(os.path.join(self.dark_paths[1], 'low'))]
        self.dark_imgs = LOL_imgs+LSRW_imgs
        self.dark_bgs = LOL_bgs+LSRW_bgs

        self.paired_transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.n_rain = len(self.rain_imgs)
        self.n_haze = len(self.haze_imgs)
        self.n_dark = len(self.dark_imgs)
    def get(self,labels):
        imgBs = []
        imgDs = []
        for index in labels:
            imgB, imgD = self.get_pair(index)
            imgBs.append(imgB.unsqueeze(0))
            imgDs.append(imgD.unsqueeze(0))
        imgBs = torch.cat(imgBs,dim=0)
        imgDs = torch.cat(imgDs, dim=0)
        return imgBs,imgDs


    def get_pair(self, index):
        if index==0: # rain
            i = np.random.randint(0,self.n_rain)
            imgD = Image.open(self.rain_imgs[i]).convert('RGB')
            imgB = Image.open(self.rain_bgs[i]).convert('RGB')
        elif index==1: # haze
            i = np.random.randint(0,self.n_haze)
            imgD = Image.open(self.haze_imgs[i]).convert('RGB')
            imgB = Image.open(self.haze_bgs[i]).convert('RGB')
        else: # dark
            i = np.random.randint(0, self.n_dark)
            imgD = Image.open(self.dark_imgs[i]).convert('RGB')
            imgB = Image.open(self.dark_bgs[i]).convert('RGB')


        if np.random.random() < 0.5:
            w, h = imgD.size
            if w>self.img_size and h>self.img_size:
                dw = w-self.img_size
                dh = h-self.img_size
                ws = np.random.randint(dw+1)
                hs = np.random.randint(dh+1)
                imgD = imgD.crop((ws, hs, ws+self.img_size, hs+self.img_size))
                imgB = imgB.crop((ws, hs, ws + self.img_size, hs + self.img_size))
        if np.random.random() < 0.5:
            imgD= imgD.transpose(Image.FLIP_LEFT_RIGHT)
            imgB = imgB.transpose(Image.FLIP_LEFT_RIGHT)

        imgD = self.paired_transform(imgD)
        imgB = self.paired_transform(imgB)
        return imgB,imgD

