import torch
import torch.nn as nn
import torch.nn.functional as F

class VoxelnetLoss(nn.Module):
    
    def __init__(self, alpha=1.5, beta=1) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smoothl1loss = nn.SmoothL1Loss(size_average=False)
        
    def forward(self, pred_reg, pred_score, pos, neg, reg):
        
        pred_reg = pred_reg.permute(0,2,3,1)
        pred_reg = pred_reg.reshape(*pred_reg.shape[:3],-1,7)
        reg = reg.reshape(*reg.shape[:3],-1,7)
        pred_score = pred_score.permute(0,2,3,1)
        pred_score = F.sigmoid(pred_score)
        where_pos = torch.where(pos==1)
        L_reg = self.smoothl1loss(pred_reg[where_pos], reg[where_pos])/pos.sum()

        pos_loss = F.binary_cross_entropy(pred_score[pos==1], pos[pos==1])
        neg_loss = F.binary_cross_entropy(1-pred_score[neg==1], neg[neg==1])
        
        L_cls = self.alpha*pos_loss + self.beta*neg_loss
        
        return L_cls, L_reg
        