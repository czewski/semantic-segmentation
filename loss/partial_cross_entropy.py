# Torch
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
# https://arxiv.org/pdf/1708.02002
class PartialCrossEntropyLoss(nn.Module): #similar to wcce?
    def __init__(self, alpha=0.25, gamma=2.0): 
        # paper says that 2 worked better for gamma, but can test with [0,5]
        super(PartialCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        #self.class_weights = class_weights

    def forward(self, pre, gt): 
        # pCE = sum (focal loss (pre, gt) * mask) / sum (mask) + 0.00001 
        # focal loss = - alpha * mod_factor * log(pt)  
        # mod_factor = (1 - pt)^gamma
        # - log(pt) = CEloss(p,y)

        #mask is gt?, or should i change to 0/1
        mask_labeled = (gt != 255).long()
        # print(pre.device, gt.device)

        logpt =  F.cross_entropy(pre, gt, reduction='none', ignore_index=255) # -1 *?
        # print(logpt.shape, logpt.device, logpt.dtype)
        # print("hi")
        pt = torch.exp(-logpt) #-
        mod_factor = torch.pow((1 - pt), self.gamma)
        focal_loss = self.alpha * mod_factor * logpt
        pCE = (focal_loss * mask_labeled).sum() / (mask_labeled.sum() + 0.0000001)

        # why use mean? 
        # https://stackoverflow.com/questions/51961275/how-is-cross-entropy-calculated-for-pixel-level-prediction
        return pCE # loss.mean() # loss.sum()