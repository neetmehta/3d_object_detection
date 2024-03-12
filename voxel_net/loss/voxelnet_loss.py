import torch
import torch.nn as nn

class VoxelnetLoss(nn.Module):
    
    def __init__(self, alpha=1.5, beta=1) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred_score, pred_reg, pos, neg, reg):
        
        pass
        
        