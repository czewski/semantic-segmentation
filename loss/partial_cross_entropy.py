# Torch
from torch import nn
import torch.nn.functional as F

class PartialCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None): # higher value to rarer classes
        super(PartialCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, pre, gt, mask): 
        # pCE = sum (focal loss (pre, gt) * mask) / sum (mask) + 0.00001 
        # focal loss = - alpha * mod_factor * log(pt)  
        # mod_factor = (1 - pt)^gamma
        # - log(pt) = CEloss(p,y)

        logpt = F.cross_entropy(pre, gt)
        mod_factor = (1 - pre) ** self.gamma
        focal_loss = - self.alpha * mod_factor * logpt
        pCE = (focal_loss * mask).sum() / mask.sum() + 0.0000001

        return pCE # loss.mean() # loss.sum()