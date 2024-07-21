import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from nystrom_attention import NystromAttention  
from models.nystrom import NystromAttention
# /home/wxy/micromamba/envs/clam/lib/python3.7/site-packages/nystrom_attention/nystrom_attention.py


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512, return_attn=False):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim // 8,
            heads = 8,
            num_landmarks = dim // 2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )
        self.return_attn = return_attn

    def forward(self, x, mask):
        if self.return_attn:
            h, attn_matrix = self.attn(self.norm(x), mask=mask, return_attn=True)
            x = x + h
            return x, attn_matrix
        else:
            h = self.attn(self.norm(x), mask=mask)
            x = x + h
            return x

        


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=1)

    def forward(self, x, H, W):
        B, P, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        # x = cnn_feat # +self.proj2(cnn_feat) # self.proj(cnn_feat)+self.proj1(cnn_feat)
        x = cnn_feat + self.proj(cnn_feat)+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x



class TransMIL(nn.Module):
    
    # only for WSI in multi-label, cannot perform instance evaluation
    
    def __init__(self, fea_dim=1024, n_classes=2, dropout=False, size_arg = "small"):
        super(TransMIL, self).__init__()
        dim_size = 512
        self.pos_layer = PPEG(dim=dim_size)
        self._fc1 = nn.Sequential(nn.Linear(fea_dim, dim_size), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_size))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=dim_size)
#         self.layer2 = TransLayer(dim=512)
        self.layer2_attn = TransLayer(dim=dim_size, return_attn=False)
        self.norm = nn.LayerNorm(dim_size)
        self._fc2 = nn.Linear(dim_size, self.n_classes)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pos_layer = self.pos_layer.to(device)
        self._fc1 = self._fc1.to(device)
#         self.cls_token = self.cls_token.to(device)
        self.layer1 = self.layer1.to(device)
        self.layer2_attn = self.layer2_attn.to(device)
        self.norm = self.norm.to(device)
        self._fc2 = self._fc2.to(device)
 
    def forward(self, h):

        # h = kwargs['data'].float() #[B, n, 1024]
        # print("=====h.orishape", h.shape)
        
        B_ori = h.shape[0]
        
        h = self._fc1(h) #[B, n, 512]
        h = h.unsqueeze(0)
        #---->pad
        # print("=====h.shape_fc1", h.shape)

        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
        # print("=====h.shape2", h.shape)

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        # print("=====h.shape_cls", h.shape)

        mask = torch.ones(1, h.shape[1]).bool().cuda()

        #---->Translayer x1
        h = self.layer1(h, mask) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        # print("=====h.layer1_ppeg", h.shape)
        
        #---->Translayer x2 
        h = self.layer2_attn(h, mask)
#         h, attn_matrix = self.layer2_attn(h, mask)
        # print("=====attn_matrix.shape", attn_matrix.shape)
        # print("=====h.layer2attn", h.shape)
        B_pad = h.shape[1]
        
        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        
        Y_prob = torch.sigmoid(logits)
#         Y_prob = F.softmax(logits, dim = 1)

        # attention padding is performed on the top
#         attn_matrix = attn_matrix[...,-B_pad:,-B_pad:]
#         attn_matrix = attn_matrix[...,:(B_ori+1),:(B_ori+1)]
#         print("=====attn_matrix.shape", attn_matrix.shape)

#         attn_matrix = torch.mean(attn_matrix[...,0,1:], dim=1) # drop the clstoken

        attn_matrix = torch.ones(self.n_classes, B_ori)
        
        return logits, Y_prob, Y_hat, attn_matrix, [0]

# if __name__ == "__main__":
#     data = torch.randn((1, 6000, 1024)).cuda()
#     model = TransMIL(n_classes=2).cuda()
#     print(model.eval())
#     results_dict = model(data = data)
#     print(results_dict)
