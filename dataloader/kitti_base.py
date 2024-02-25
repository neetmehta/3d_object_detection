from typing import Any
import torch
import numpy as np
import os
from torch.utils.data import Dataset
import glob
import cv2
from collections import namedtuple

from lidar_utils.lidar_utils import (read_point_cloud, read_config, filter_point_cloud, visualize_pc, 
                                    point_cloud_to_tensor, voxelize, read_label, label_2_bb, 
                                    read_calib_file, visualize_bb, visualize_bb_from_file, point_cloud_to_tensor)

Label = namedtuple('Label', ['type', 'h', 'w', 'l', 'x', 'y', 'z', 'theta'])

def collate_fn(data):
    batch_tensor, batch_voxels, batch_zero_pos = zip(*data)
    batch_tensor = torch.cat(batch_tensor, dim=0)
    batch_zero_pos = torch.cat(batch_zero_pos, dim=0)
    
    for i in range(len(batch_voxels)):
        batch_voxels[i][:, 0] = batch_voxels[i][:, 0] * i
        
    batch_voxels = torch.cat(batch_voxels, dim=0)
    
    return batch_tensor, batch_voxels, batch_zero_pos
    

class KittiBase(Dataset):

    def __init__(self, root, config_path, training=True) -> None:
        super().__init__()
        self.root = root
        self.image_dir = os.path.join(self.root, 'image_2')
        self.pc_dir = os.path.join(self.root, 'velodyne')
        self.calib = os.path.join(self.root, 'calib')
        self.label = os.path.join(self.root, 'label_2')
        self.cfg = read_config(config_path)
        self.image_list = glob.glob(f'{self.image_dir}/*')
        self.pc_list = glob.glob(f'{self.pc_dir}/*')
        self.calib_list = glob.glob(f'{self.calib}/*')
        self.label_list = glob.glob(f'{self.label}/*')

    def __len__(self):
        assert len(self.image_list) == len(self.pc_list) == len(self.calib_list) == len(self.label_list), 'corrupted data'
        return len(self.image_list)
    
    def _parse_label(self,file_path):
        label_list = []
        with open(file_path, 'r') as f:
            label_txt = f.readlines()

        for i in label_txt:
            label_fields = i.split()
            if label_fields[0] == 'DontCare':
                continue
            kitti_object, _, _, _, _, _, _, _, h, w, l, x, y, z, theta = label_fields
            label = Label(kitti_object, h, w, l, x, y, z, theta)
            label_list.append(label)
        
        return label_list
    
    def _parse_calib_file(self, calib_path):
        calib_data = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if ':' in line :
                    key, value = line.split(':', 1)
                    calib_data[key] = np.array([float(x) for x in value.split()])

    def __getitem__(self, index: Any) -> Any:
        sample = {}
        image_path = self.image_list[index]
        label_path = self.label_list[index]
        pc_path = self.pc_list[index]
        calib_path = self.calib_list[index]

        image = cv2.imread(image_path)
        
        label_list = self._parse_label(label_path)

        point_cloud = read_point_cloud(pc_path)
        point_cloud = filter_point_cloud(self.cfg, point_cloud)
        tensor, voxels = point_cloud_to_tensor(self.cfg, point_cloud)
        voxels = torch.cat((torch.ones(len(voxels), 1, dtype=torch.int32), voxels), dim=-1)
        zero_pos = tensor.sum(-1)
        zero_pos = zero_pos==0
        # tensor = [tensor]
        # voxels = [voxels]
        # zero_pos = [zero_pos]
        return tensor, voxels, zero_pos
    
    

        

    
        