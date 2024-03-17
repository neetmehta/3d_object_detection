import torch

from dataloader.kitti import KittiBase, collate_fn
from voxel_net.network.voxelnet import Voxelnet
import torch.optim as optim
from torch.utils.data import DataLoader
from lidar_utils.lidar_utils import read_config
from losses.voxelnet_loss import VoxelnetLoss
from torch.optim import SGD

torch.autograd.set_detect_anomaly(True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg_path = r'E:\Deep Learning Projects\3d_object_detection\configs\train_config.yaml'
dataset = KittiBase(r"E:\Deep Learning Projects\datasets\training", cfg_path)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)

cfg = read_config(cfg_path)
model = Voxelnet(cfg).to(device)
criteria = VoxelnetLoss()

optimizer = SGD(model.parameters(), lr=0.01)

for idx, (batch_tensor, batch_voxels, batch_zero_pos, batch_targets) in enumerate(dataloader):
    
    pos, neg, reg = batch_targets
    batch_tensor, batch_voxels, batch_zero_pos, pos, neg, reg = batch_tensor.to(device), batch_voxels.to(device), batch_zero_pos.to(device), pos.to(device), neg.to(device), reg.to(device)
    optimizer.zero_grad()
    pred_score, pred_reg = model(batch_tensor, batch_voxels, batch_zero_pos)
    print('here')
    l_cls, l_reg = criteria(pred_reg, pred_score, pos, neg, reg)
    loss = l_cls + l_reg
    loss.backward()
    optimizer.step()
    
    print(idx, loss)
    