import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.parser import parse_args, load_config

from models import get_vit_base_patch16_224
from torchvision.models import resnet18, resnet50, resnet34
from einops import rearrange

import numpy as np
from nystrom_attention import NystromAttention

from functools import partial
import numpy as np
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_



class DepResLoss(nn.Module):
    def __init__(self, target_number = None):
        super(DepResLoss, self).__init__()
         #####
        
        self.target_number = target_number
        self.resnet18 = resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-2])
        
        self.resnet34 = resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.resnet34 = nn.Sequential(*list(self.resnet34.children())[:-2])

        self.resnet50 = resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])

        self.L = 768
        self.D = 128
        self.K = 1


        self.conv_fuse = nn.Sequential(
            nn.Conv2d(512, 768, 3),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        if self.target_number in ['WS', 'CH', 'RAC']:
            t = 3
        else: 
            t = 2

        self.linear = nn.Linear(self.L*self.K, t)
 

        self.deplinear1 = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU()
        )
        self.deplinear2 = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU()
        )
        self.deplinear3 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU()
        )
        self.deplinear4 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L + 128, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        opt = parse_args()
        opt.cfg_file = "/data1/ryqiu/Zhongshan/EndoMIL/models/configs/Kinetics/TimeSformer_divST_8x32_224.yaml"
        config = load_config(opt)
        backbone = get_vit_base_patch16_224(cfg=config)

        pretrained_weights = '/data1/ryqiu/Zhongshan/EndoMIL/endo_fm.pth'
        ckpt = torch.load(pretrained_weights, map_location='cpu')
        if "teacher" in ckpt:
            ckpt = ckpt["teacher"]
        renamed_checkpoint = {x[len("backbone."):]: y for x, y in ckpt.items() if x.startswith("backbone.")}
        msg = backbone.load_state_dict(renamed_checkpoint, strict=False)
        # print(f"Loaded model with msg: {msg}")

        backbone = backbone.cuda()
        self.endo = backbone

        for p in self.endo.parameters():
            p.requires_grad = False
        
        self.gap = nn.AdaptiveAvgPool2d((1,1)) 

        self.align = nn.Linear(512,768)

        self.align_nmiV2 = nn.Linear(2048,768)
        # PureDep ************ 
        self.attention_pure = nn.Sequential(
            nn.Linear(512+1, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.linear_pure = nn.Linear(512, t)
        # PureDep ************ 

        # abmil ************ 
        self.attention_abmil = nn.Sequential(
            nn.Linear(512, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.linear_abmil = nn.Linear(512, t)
        # abmil ************ 

        # codeddep ************ 
        self.attention_codeddep = nn.Sequential(
            nn.Linear(512+128, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.linear_codeddep = nn.Linear(640, t)
        # # codeddep ************ 

        # endo ************ 
        self.attention_endo = nn.Sequential(
            nn.Linear(768, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.linear_endo = nn.Linear(768, t)
        # endo ************ 

        # endodep ************ 
        self.linear_endodep = nn.Linear(768+128, t)
        # endodep ************ 


    def _forward_ours(self, x, bag_dep, flag=1):
        '''
        大模型约束，深度信息
        '''
        X = rearrange(x, 't b c h w -> b c t h w ')
        X = self.endo(X)
        X = X[:,1:]
        X = X.mean(dim=1)

        H = self.resnet18(x.squeeze(0))  # [N,512,16,16]
        H = self.gap(H)
        H = H.squeeze()
        H = self.align(H)

        bag_dep_temp = (bag_dep - min(bag_dep))/(max(bag_dep) - min(bag_dep)) #scaling

        alpha = self.deplinear1(torch.transpose(bag_dep_temp.unsqueeze(0),1,0)) # NxC
        alpha = self.deplinear2(alpha)
        alpha = self.deplinear3(alpha)
        alpha = self.deplinear4(alpha)
        A = self.attention(torch.cat((H, alpha), dim=1))  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL
        y = self.linear(M)

        return y, A, M, X, H #pred, attn vextor, final matrix, endo_output, r
   
    def forward(self, x, bag_dep):

        return self._forward_ours(x, bag_dep)

        
