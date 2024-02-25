import torch
import torch.nn as nn
from .layers import StackedVFE, SparseMiddleEncoder, RPN
from mmcv.ops.sparse_structure import SparseConvTensor

class Voxelnet(nn.Module):
    
    def __init__(self, cfg) -> None:
        super().__init__()
        
        x_min = cfg['x_min']
        x_max = cfg['x_max']

        y_min = cfg['y_min']
        y_max = cfg['y_max']

        z_min = cfg['z_min']
        z_max = cfg['z_max']

        vd = cfg['vd']
        vh = cfg['vh']
        vw = cfg['vw']
        
        self.W = int((x_max - x_min)/(vw))
        self.H = int((y_max - y_min)/(vh))
        self.D = int((z_max - z_min)/(vd))
        self.C = cfg['vfe_out_channels']
        self.stacked_vfe = StackedVFE(cfg['vfe_layers'], cfg['vfe_out_channels'])
        self.sparse_encoder = SparseMiddleEncoder(cfg['sparse_encoder_layers'])
        self.rpn = RPN(cfg['rpn_conv_block'], cfg['rpn_deconv_block'])
        
    def forward(self, x, voxels, zero_pos):
        
        x = self.stacked_vfe(x, zero_pos)
        sparse_tensor = SparseConvTensor(x, voxels[:, [0,3,2,1]], (self.D, self.H, self.W), torch.max(voxels[:, 0], dim=0)[0].item()+1)
        sparse_tensor = self.sparse_encoder(sparse_tensor)
        feats = sparse_tensor.dense()
        score, reg = self.rpn(feats)
        return score, reg        