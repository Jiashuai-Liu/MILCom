import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
from random import random, randint

class NICWSS(nn.Module):
    def __init__(self, n_classes=2, fea_dim=1024, dropout=True, size_arg='small'):
        nn.Module.__init__(self)

        self.size_dict = {"small": [fea_dim, 64], "big": [fea_dim, 128]}  # 128 for nic, 64 for nicwss
        size = self.size_dict[size_arg]
        
        conv_nic = [nn.Conv2d(size[0], size[1], kernel_size=3, stride=1, padding=1, bias=False)] #  dilation=1
        conv_nic.append(nn.BatchNorm2d(size[1]))
        conv_nic.append(nn.ReLU())
        if dropout:
            conv_nic.append(nn.Dropout(0.25)) # dropout little better than no dropout & dropout2d 
        self.conv_nic = nn.Sequential(*conv_nic)
        
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip([6, 12, 18, 24], [6, 12, 18, 24]):
            self.conv2d_list.append(
                nn.Conv2d(
                    size[1],  # 128
                    n_classes,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=False,
                )
            )

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)
        
        # self.conv11 = torch.nn.Conv2d(size[1], size[1],  1, bias=False)
        
        # theta PCM
        self.pcm_head = torch.nn.Conv2d(size[1], size[1],  1, bias=False) # PCM 1x1conv
        torch.nn.init.xavier_uniform_(self.pcm_head.weight)

        self.n_classes = n_classes
        
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_nic = self.conv_nic.to(device)
        self.conv2d_list = self.conv2d_list.to(device)
        self.pcm_head = self.pcm_head.to(device)
        # self.conv11 = self.conv11.to(device)
        
    # def max_onehot(self, x, x_max=0.2):
    #     x[:, :, :, :][x[:, :, :, :] < x_max]  = 0.0
    #     x[:, :, :, :][x[:, :, :, :] >= x_max] = 0.9
    #     max_values = torch.max(x, dim=1, keepdims=True)
    #     one_hot = torch.zeros_like(x)
    #     one_hot[x == max_values] = 1.0
    #     return one_hot.long()  # argmax over c dimension
    
    def max_onehot(self, x, x_max=0.2):
        x[:, :, :, :][x[:, :, :, :] < x_max]  = 0.0
        x[:, :, :, :][x[:, :, :, :] >= x_max] = 0.9
        return x.argmax(1).long()
    
    # def max_onehot(self, x, x_max=0.2):
    #     x[:, :, :, :][x[:, :, :, :] < x_max]  = 0
    #     x[:, :, :, :][x[:, :, :, :] >= x_max] = 1
    #     return torch.max(x, dim=1)[0].long()


    def PCM(self, cam, f, size=(128,128), out_size=(256,256)):
        f = self.pcm_head(f)
        cam = F.interpolate(cam, size=size, mode='bilinear', align_corners=True) # 奇怪的下采样
        f   = F.interpolate(f,   size=size, mode='bilinear', align_corners=True)
        n, c, h, w = cam.shape
        cam = cam.view(n, -1, h * w)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True)+ 1e-5)
        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff / (torch.sum(aff, dim=1, keepdim=True)+ 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)
        cam_rv = F.interpolate(cam_rv, size=out_size, mode='bilinear', align_corners=True)
        return cam_rv


    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, 1)

    @staticmethod
    def predictions(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(1)

    @staticmethod
    def loss(logits: torch.Tensor, labels: torch.Tensor):
        return F.multilabel_soft_margin_loss(logits, labels)

    def forward(self, h, input_size=(128,128), out_size=(256,256), masked=True, train=False):


        device = h.device
        
        C, H, W = h.shape

        h = self.conv_nic(h.unsqueeze(0))
        # h = F.interpolate(h,size=(256,256), mode='bilinear', align_corners=True)
        # 进ASPP前是否变成正方形
        cam = self.conv2d_list[0](h)   # ASPP
        for i in range(len(self.conv2d_list) - 1):
            cam += self.conv2d_list[i + 1](F.leaky_relu(h))  # add relu?
        
        return cam