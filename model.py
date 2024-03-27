
import sys
sys.path.append("./../")
import os
import numpy as np
import random
from torch import einsum
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.utils.data as dataf
from torch.utils.data import Dataset
from scipy import io
from scipy.io import loadmat as loadmat
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
from torch.nn import LayerNorm,Linear,Dropout,Softmax
import time
from PIL import Image
import math
from operator import truediv
from torchsummary import summary
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import re
from pathlib import Path
import copy

import utils
import torch.nn as nn
from torch.nn import init
import transformer
from transformer1 import HetConv,Encoder

from einops import rearrange, repeat


def Init_Weights(net, init_type, gain):
    print('Init Network Weights')
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 or classname.find('BatchNorm1d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class MUNet(nn.Module):
    def __init__(self, band, num_classes, ldr_dim, reduction):
        super(MUNet, self).__init__()
        self.classes=num_classes
        self.fc_hsi = nn.Sequential(
            nn.Conv2d(band, band//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(band//2),
            nn.ReLU(),
            nn.Conv2d(band//2, band//4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(band//4),
            nn.ReLU(),
            nn.Conv2d(band//4, num_classes, kernel_size=1, stride=1, padding=0)
        )
        
        self.spectral_fe = nn.Sequential(
            nn.Conv2d(ldr_dim, num_classes//reduction, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_classes//reduction),
            nn.ReLU(),
            nn.Conv2d(num_classes//reduction, num_classes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
            nn.Conv2d(num_classes, 1, kernel_size=1, stride=1, padding=0)
        )
        
        
        self.spectral_se = nn.Sequential(
            nn.Conv2d(num_classes, num_classes//reduction, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(num_classes//reduction, num_classes, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=1)

        self.decoder = nn.Sequential(
            nn.Conv2d(num_classes, band, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
        )
        self.upscale = nn.Sequential(
            nn.Linear(num_classes, band),
        )
        # self.vtrans = transformer.ViT(image_size=1, patch_size=1, dim=num_classes*2, depth=2,
        #                               heads=5, mlp_dim=12, pool='cls')
        self.ca = Encoder(band)
        self.dropout = nn.Dropout(0.1)
        self.fclayer = nn.Linear(band , num_classes)
        
        self.clsTok = nn.Parameter(torch.zeros(1, 1, band))

        
        
    def forward(self, x, y):
        encode=self.fc_hsi(x)
        ## spectral attention
        y_fe = self.spectral_fe(y)

        encode=encode.view(encode.shape[0],1 ,-1)
        encode=self.upscale(encode)
        
        attention=self.ca(encode,y_fe)

        attention = self.fclayer(attention)
        attention = attention.view(attention.shape[0],self.classes ,1,1)
        abu = self.softmax(attention)
        
        output = self.decoder(abu)
        
        return abu, output

        # encode = self.fc_hsi(x)
        # ## spectral attention
        # y_fe = self.spectral_fe(y)

        # attention = self.spectral_se(y_fe)
        # abu = self.softmax(torch.mul(encode, attention))
    
        # output = self.decoder(abu)

        # return abu, output
    

class AutoEncoder(nn.Module):
    def __init__(self, P, L, size, patch, dim,ldr_dim,reduction):
        super(AutoEncoder, self).__init__()
        self.P, self.L, self.size, self.dim = P, L, size, dim
        num_classes=P
        band=L

        self.fc_hsi = nn.Sequential(
            nn.Conv2d(band, band//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(band//2,momentum=0.9),
            nn.Dropout(0.25),
            nn.LeakyReLU(),
            nn.Conv2d(band//2, band//4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(band//4,momentum=0.9),
            nn.LeakyReLU(),
            nn.Conv2d(band//4, num_classes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_classes,momentum=0.5),
        )
        
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(L, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        #     nn.BatchNorm2d(128, momentum=0.9),
        #     nn.Dropout(0.25),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        #     nn.BatchNorm2d(64, momentum=0.9),
        #     nn.LeakyReLU(),
        #     nn.Conv2d(64, (dim*P)//patch**2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        #     nn.BatchNorm2d((dim*P)//patch**2, momentum=0.5),
        # )

        self.vtrans = transformer.ViT(image_size=size, patch_size=patch, dim=num_classes, depth=2,
                                      heads=5, mlp_dim=12, pool='cls')
        
        
        # self.vtrans = transformer.ViT(image_size=size, patch_size=patch, dim=(dim*P), depth=2,
        #                               heads=8, mlp_dim=12, pool='cls')
        
        self.upscale = nn.Sequential(
            # nn.Linear(dim, size * 130),
            nn.Linear(1, size * 130),

        )
        self.ca = TransformerEncoder(130)
        self.smooth = nn.Sequential(
            nn.Conv2d(P, P, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Softmax(dim=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(P, L, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )
        self.spectral_fe = nn.Sequential(
            nn.Conv2d(ldr_dim, num_classes//reduction, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_classes//reduction),
            nn.ReLU(),
            nn.Conv2d(num_classes//reduction, num_classes, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
        )
        self.spectral_se = nn.Sequential(
            nn.Conv2d(num_classes, num_classes//reduction, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(num_classes//reduction, num_classes, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=1)
        self.lidarConv = nn.Sequential(
                        nn.Conv2d(band,64,3,1,1),
                        nn.BatchNorm2d(64),
                        nn.GELU()
                        )


    @staticmethod
    def weights_init(m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x,y):
        abu_est = self.fc_hsi(x)
        y_fe = self.spectral_fe(y)

        abu_est=abu_est.reshape(1,abu_est.shape[1],abu_est.shape[0]).cuda()
        y_fe=y_fe.reshape(1,y_fe.shape[1],y_fe.shape[0]).cuda()

        abu_est = torch.cat((abu_est, y_fe), dim = 1) 
        print("BBBBBB")
        print(abu_est.shape)
        print("BBBBB")
        # cls_emb = self.vtrans(abu_est)
        cls_emb = self.ca(abu_est)
        
        print("AAAAAA")
        print(cls_emb.shape)
        print("AAAAAA")

        # cls_emb = cls_emb.view(1, self.P, -1)
        # abu_est = self.upscale(cls_emb).view(1, self.P, self.size, 130)
        # abu_est = cls_emb.reshape(1, self.P, self.size, 130)
        abu_est = self.smooth(cls_emb)


        abu_est = self.softmax(torch.mul(abu_est, cls_emb))
        re_result = self.decoder(abu_est)
        return abu_est, re_result

        # cls_emb = cls_emb.view(1, self.P, -1)
        # abu_est = self.upscale(cls_emb).view(1, self.P, self.size, 130)
        # abu_est = self.smooth(abu_est)
        
        # # attention = self.spectral_se(y_fe)
        
        # abu_est = self.softmax(torch.mul(abu_est, y_fe))
        # re_result = self.decoder(abu_est)
        # return abu_est, re_result

class MFT(nn.Module):
    def __init__(self, FM, NC, NCLidar, Classes, HSIOnly):
        
        super(MFT, self).__init__()
        self.HSIOnly = HSIOnly
        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0,1,1), stride = 1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        
        self.conv6 = nn.Sequential(
            HetConv(8 * (NC - 8), FM*4,
                p = 1,
                g = (FM*4)//4 if (8 * (NC - 8))%FM == 0 else (FM*4)//8,
                   ),
            nn.BatchNorm2d(FM*4),
            nn.ReLU()
        )
        
        self.last_BandSize = NC//2//2//2
        
        self.lidarConv = nn.Sequential(
                        nn.Conv2d(NCLidar,64,3,1,1),
                        nn.BatchNorm2d(64),
                        nn.GELU()
                        )
        self.decoder = nn.Sequential(
            nn.Conv2d(Classes, NC, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.ReLU(),
        )
        self.ca = TransformerEncoder(FM*4)
        self.out3 = nn.Linear(FM*4 , Classes)
        self.position_embeddings = nn.Parameter(torch.randn(1, 4 + 1, FM*4))
        self.dropout = nn.Dropout(0.1)
        torch.nn.init.xavier_uniform_(self.out3.weight)
        torch.nn.init.normal_(self.out3.bias, std=1e-6)
        self.token_wA = nn.Parameter(torch.empty(1, 4, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)
        
        self.token_wA_L = nn.Parameter(torch.empty(1, 1, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA_L)
        self.token_wV_L = nn.Parameter(torch.empty(1, 64, 64),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV_L)
                
        

    def forward(self, x1, x2):

        print([x1.shape,x2.shape])
        x1 = x1.reshape(x1.shape[0],-1,patchsize,patchsize)
        x1 = x1.unsqueeze(1)
        x2 = x2.reshape(x2.shape[0],-1,patchsize,patchsize)
        x1 = self.conv5(x1)
        x1 = x1.reshape(x1.shape[0],-1,patchsize,patchsize)
        
        x1 = self.conv6(x1)
        x2 = self.lidarConv(x2)


        x2 = x2.reshape(x2.shape[0],-1,patchsize**2)
        x2 = x2.transpose(-1, -2)
        wa_L = self.token_wA_L.expand(x1.shape[0],-1,-1)
        wa_L = rearrange(wa_L, 'b h w -> b w h')  # Transpose
        A_L = torch.einsum('bij,bjk->bik', x2, wa_L)
        A_L = rearrange(A_L, 'b h w -> b w h')  # Transpose
        A_L = A_L.softmax(dim=-1)
        wv_L = self.token_wV_L.expand(x2.shape[0],-1,-1)
        VV_L = torch.einsum('bij,bjk->bik', x2, wv_L)
        x2 = torch.einsum('bij,bjk->bik', A_L, VV_L)
        x1 = x1.flatten(2)
        
        x1 = x1.transpose(-1, -2)
        wa = self.token_wA.expand(x1.shape[0],-1,-1)
        wa = rearrange(wa, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x1, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)
        wv = self.token_wV.expand(x1.shape[0],-1,-1)
        VV = torch.einsum('bij,bjk->bik', x1, wv)
        T = torch.einsum('bij,bjk->bik', A, VV)
        x = torch.cat((x2, T), dim = 1) #[b,n+1,dim]
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        x = self.ca(embeddings)
        x = x.reshape(x.shape[0],-1)
        out3 = self.out3(x)
        return out3
  
