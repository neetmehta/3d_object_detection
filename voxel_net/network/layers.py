import torch
import torch.nn as nn
from mmcv.ops import SparseSequential, SparseConv3d

class LinearBNReLU(nn.Module):
    
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn(self.linear(x)))
        
        return x
    
class VFE(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        assert out_channels%2==0, 'out_channels must be even number'
        self.lin_bn_relu = LinearBNReLU(in_channels=in_channels, out_channels=out_channels//2)
        
        
    def forward(self, x, zero_pos):
        
        K, T, in_channels = x.shape
        x = x.view(-1, in_channels)
        x = self.lin_bn_relu(x)
        x = x.view(K, T, -1)
        x[zero_pos] = 0
        aug = torch.max(x, dim=1, keepdim=True)[0]
        aug = aug.repeat(1, T, 1)
        x = torch.cat([x, aug], dim=-1)
        x[zero_pos] = 0
        return x
        

class StackedVFE(nn.Module):
    
    def __init__(self, vfe_layers, out_channels) -> None:
        super().__init__()
        
        vfe = []
        for in_ch, out_ch in vfe_layers:
            vfe.append(self._make_vfe_layer(in_ch, out_ch))
            
        self.stacked_vfe = nn.ModuleList(vfe)
        self.fcn = LinearBNReLU(vfe_layers[-1][-1], out_channels)
        
    def _make_vfe_layer(self, in_ch, out_ch):
        return VFE(in_ch, out_ch)
        
    def forward(self, x, zero_pos):
        
        for vfe in self.stacked_vfe:
            x = vfe(x, zero_pos)
            
        K, T, C = x.shape
        x = x.view(-1, C)
        x = self.fcn(x)
        x = x.view(K, T, -1)
        x = torch.max(x, dim=1)[0]
        return x
    
    
class SparseMiddleEncoder(nn.Module):
    
    def __init__(self, layers) -> None:
        super().__init__()
        
        cml = []
        
        for in_ch, out_ch, k, s, p in layers:
            cml.append(self._make_spconv_bn_relu(in_ch, out_ch, k, s, p))
            
        self.cml = nn.Sequential(*cml)
        
    
    def _make_spconv_bn_relu(self, in_ch, out_ch, k, s, p):
        
        return SparseSequential(
            SparseConv3d(in_ch, out_ch, k, s, p),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
            )
        
        
    def forward(self, sparse_tensor):

        return self.cml(sparse_tensor)
    
class RPN(nn.Module):
    
    def __init__(self, rpn_conv_block, rpn_deconv_block) -> None:
        super().__init__()
        
        self.rpn_conv_block = nn.ModuleList()
        self.rpn_deconv_block = nn.ModuleList()
        
        for block in rpn_conv_block:

            self.rpn_conv_block.append(self._make_rpn_block(block))
        
        out_feats = 0    
        for block in rpn_deconv_block:

            self.rpn_deconv_block.append(self._make_rpn_block(block))
            out_feats += block[0][1][0][1]
            
        self.reg = nn.Conv2d(out_feats, 14, 1, 1, 0)
        self.score = nn.Conv2d(out_feats, 2, 1, 1, 0)
        
    def _make_conv_bn_relu(self, in_ch, out_ch, k, s, p):
        
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
            )
    def _make_deconv(self, in_ch, out_ch, k, s, p):
        return nn.ConvTranspose2d(in_ch, out_ch, k, s, p)
        
    def _make_rpn_block(self, block):
        
        _block = []
        for block_type, [[in_ch, out_ch, k, s, p], n] in block:
            
            for _ in range(n):
                if block_type == 'conv':
                    _block.append(self._make_conv_bn_relu(in_ch, out_ch, k, s, p))
                    
                elif block_type == 'deconv':
                    _block.append(self._make_deconv(in_ch, out_ch, k, s, p))
                
        return nn.Sequential(*_block)
    
    def forward(self, x):
        
        B, C, D, H, W = x.shape
        x = x.view(B, C*D, H, W)
        deconv_feats = []
        for conv_layer, deconv_layer in zip(self.rpn_conv_block, self.rpn_deconv_block):
            x = conv_layer(x)
            deconv = deconv_layer(x)
            deconv_feats.append(deconv)
            
        feats = torch.cat(deconv_feats[::-1], dim=1)
            
        return self.score(feats), self.reg(feats)
            
            

