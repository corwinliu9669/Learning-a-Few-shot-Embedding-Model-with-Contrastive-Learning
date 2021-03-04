#### This file is based on code of cross attention network
from resnet import resnet12
from utils import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, scale_cls=7, num_classes=64):
        super(Model, self).__init__()
        self.scale_cls = scale_cls
        self.base = resnet12()
        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1) 

    def test(self, ftrain, ftest):
        ftest = ftest.mean(4)
        ftest = ftest.mean(4)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores
    
    def reform(self, f1, f2):
        b, n1, c, h, w = f1.size()
        n2 = f2.size(1)
        f1 = f1.view(b, n1, c, -1) 
        f2 = f2.view(b, n2, c, -1)
        f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)
        f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)
        f1_norm = f1_norm.transpose(2, 3).unsqueeze(2) 
        f2_norm = f2_norm.unsqueeze(1)
        a1 = torch.matmul(f1_norm, f2_norm) 
        a2 = a1.transpose(3, 4) 
        f1 = f1.unsqueeze(2)
        f1 = f1.repeat(1,1,n2,1,1)
        f1 = f1.view(b, n1, n2, c, h, w)
        f2 = f2.unsqueeze(1)
        f2 = f2.repeat(1, 1, n1, 1, 1)
        f2 = f2.view(b, n1, n2, c, h, w)
        return f1.transpose(1, 2), f2.transpose(1, 2)

    def forward(self, xtrain, xtest, ytrain, ytest):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)
        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)
        global_cls = self.clasifier(f)
        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1) 
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])
        ftest = f[batch_size * num_train:]
        if self.training:
            ftrain = random_block(ftrain)
        ftest = ftest.view(batch_size, num_test, *f.size()[1:]) 
        ftrain, ftest = self.reform(ftrain, ftest)
        ftrain = ftrain.mean(4)
        ftrain = ftrain.mean(4)
        if not self.training:
            return self.test(ftrain, ftest)
        ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)
        cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
        cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])
        return global_cls, cls_scores
