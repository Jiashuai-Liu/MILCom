import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import numpy as np
from random import random, randint

class NICWSS(nn.Module):
    def __init__(self, n_classes=2, fea_dim=1024, dropout=True, size_arg='small', only_cam=True):
        nn.Module.__init__(self)

        self.size_dict = {"small": [fea_dim, 64], "big": [fea_dim, 128]}  # 128 for nic, 64 for nicwss
        size = self.size_dict[size_arg]
        self.only_cam = only_cam
        
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
                    size[1],  # decided by the size_dict
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
        
        
        if not self.only_cam:
            # theta PCM
            self.pcm_head = torch.nn.Conv2d(size[1], size[1],  1, bias=False) # PCM 1x1conv
            torch.nn.init.xavier_uniform_(self.pcm_head.weight)
            self.conv11 = torch.nn.Conv2d(size[1], size[1],  1, bias=False)


        self.n_classes = n_classes
        
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_nic = self.conv_nic.to(device)
        self.conv2d_list = self.conv2d_list.to(device)
        if not self.only_cam:
            self.pcm_head = self.pcm_head.to(device)
            self.conv11 = self.conv11.to(device)
        
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
        
        if self.only_cam:
            return cam
        else:
            n, c, height, width = cam.shape
            
            # with torch.no_grad():
            # cam_d = F.relu(cam.detach())
            cam_d = cam.detach()
            cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1) + 1e-5  # n, c, 1, 1
            # print(f" the cam_d_max.shape is {cam_d_max.shape}")  # for test
            cam_d_norm = (cam_d.clone() - 1e-5) / cam_d_max  # n, c, H, W  cam Normalize
            ############
            # for multi-class and multi-label
            cam_rv = []
            for cls in range(self.n_classes):
                cam_d_norm_cls = torch.zeros((n, 2, height, width)).to(device)
                cam_d_norm_cls[:, 0, :, :] = 1 - cam_d_norm[:, cls, :, :]
                cam_d_norm_cls[:, 1, :, :] = cam_d_norm[:, cls, :, :]
                cam_max = torch.max(cam_d_norm_cls, dim=1, keepdim=True)[0]
                cam_d_norm_cls[cam_d_norm_cls < cam_max] = 0  # n, 2, H, W
                # cam_d_norm[:, 0, :, :] = 1 - torch.max(cam_d_norm[:, 1:, :, :], dim=1)[0]
                # cam_max = torch.max(cam_d_norm, dim=1, keepdim=True)[0]  # n, 1, H, W
                # cam_d_norm[cam_d_norm < cam_max] = 0 # n, c, H, W
            
                if masked:
                    max_ones = []
                    thr_list = [0.2, 0.3, 0.4]
                    # thr_list = [0.1, 0.2]
                    for i in thr_list:
                        max_ones.append(self.max_onehot(cam_d_norm_cls.clone(), x_max=i).unsqueeze(1))
                    max_onehot = torch.mean(torch.cat(max_ones, 1).float(), dim=1)
                    max_onehot = max_onehot.unsqueeze(1)
                    #  print(max_onehot.repeat(1,h.shape[1],1,1))
                else:
                    max_onehot = torch.ones((n, height, width)).long().cuda().unsqueeze(1)
                    
                h_cls = h * max_onehot
                # 进PCM前有个conv
                h_cls = self.conv11(h_cls)
                # cam_rv_cls = self.PCM(cam_d_norm_cls, h_cls.detach(), input_size, (H,W))
                cam_rv_cls = self.PCM(cam_d_norm_cls, h_cls, input_size, (H,W))
                # cam_rv_cls = F.interpolate(self.PCM(cam_d_norm_cls, h.detach(), input_size, out_size),
                #                     (H, W), mode='bilinear', align_corners=True)
                cam_rv.append(cam_rv_cls[:, 1, :, :].unsqueeze(1))
            cam_rv = torch.cat(cam_rv, dim=1)
            ############
            # cam_rv = F.interpolate(torch.cat(cam_rv, dim=1), (H, W), mode='bilinear', align_corners=True)
            # cam = F.interpolate(cam, (H, W), mode='bilinear', align_corners=True)
            # max_onehot = F.interpolate(max_onehot, (H, W), mode='bilinear', align_corners=True)
            
            if train:
                return cam, cam_rv, max_onehot
            else:
                return cam, cam_rv