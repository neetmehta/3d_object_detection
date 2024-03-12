from typing import Any
import torch
import numpy as np
import os
from torch.utils.data import Dataset
import glob
import cv2
from collections import namedtuple
import random
# from mmcv.ops.bbox import bbox_overlaps
from mmdet.structures.bbox import bbox_overlaps

from lidar_utils.lidar_utils import (read_point_cloud, read_config, filter_point_cloud, visualize_pc, 
                                    point_cloud_to_tensor, voxelize, read_label, label_2_bb, 
                                    read_calib_file, visualize_bb, visualize_bb_from_file, point_cloud_to_tensor, ry_to_rz, box2corner)

Label = namedtuple('Label', ['type', 'h', 'w', 'l', 'x', 'y', 'z', 'theta'])

def collate_fn(data):
    batch_tensor, batch_voxels, batch_zero_pos, batch_label, batch_calib, batch_targets = zip(*data)
    pos, neg, reg = [], [], []
    
    for target in batch_targets:
        p, n, r = target
        pos.append(p)
        neg.append(n)
        reg.append(r)
        
    batch_targets = (torch.stack(pos, dim=0), torch.stack(neg, dim=0), torch.stack(reg, dim=0))
    batch_tensor = torch.cat(batch_tensor, dim=0)
    batch_zero_pos = torch.cat(batch_zero_pos, dim=0)
    
    for i in range(len(batch_voxels)):
        batch_voxels[i][:, 0] = batch_voxels[i][:, 0] * i
        
    batch_voxels = torch.cat(batch_voxels, dim=0)
    
    return batch_tensor, batch_voxels, batch_zero_pos, batch_label, batch_calib, batch_targets
    

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
        self.create_anchors()

    def __len__(self):
        assert len(self.image_list) == len(self.pc_list) == len(self.calib_list) == len(self.label_list), 'corrupted data'
        return len(self.image_list)
    
    def create_anchors(self):
        x_min = self.cfg['x_min']
        x_max = self.cfg['x_max']

        y_min = self.cfg['y_min']
        y_max = self.cfg['y_max']


        vh = self.cfg['vh']     # Y
        vw = self.cfg['vw']  

        x_start = x_min + vw
        x_end = x_max - vw
        y_start = y_min + vh
        y_end = y_max - vh

        c_xy = torch.stack(torch.meshgrid(torch.linspace(x_start, x_end, int((x_max - x_min)/(2*vw))), torch.linspace(y_start, y_end, int((y_max - y_min)/(2*vh))), indexing='xy'))
        c_zhwl = torch.ones((4, c_xy.shape[1], c_xy.shape[2]))
        c_theta_0 = torch.ones((1, c_xy.shape[1], c_xy.shape[2]))*0
        c_theta_90 = torch.ones((1, c_xy.shape[1], c_xy.shape[2]))*(torch.pi/2)
        c_zhwl[0, ...] = c_zhwl[0, ...]*-1
        c_zhwl[3, ...] = c_zhwl[3, ...]*3.9
        c_zhwl[2, ...] = c_zhwl[2, ...]*1.6
        c_zhwl[1, ...] = c_zhwl[1, ...]*1.56

        c = torch.cat((c_xy, c_zhwl, c_theta_0, c_xy, c_zhwl, c_theta_90), dim=0)
        self.anchors = c.permute(1,2,0).reshape(-1,7)
        
        
    def _parse_label(self, file_path, calib_path):
        
        tr_velo_2_cam = read_calib_file(calib_path)['Tr_velo_to_cam']
        T = np.zeros((4,4))
        T[:3,:] = tr_velo_2_cam
        T[3,3] = 1
        label_list = []
        
        with open(file_path, 'r') as f:
            label_txt = f.readlines()

        for i in label_txt:
            label_fields = i.split()
            kitti_object, _, _, _, _, _, _, _, h, w, l, x, y, z, theta = label_fields
            if kitti_object not in self.cfg['classes']:
                continue
            
            h, w, l, x, y, z, theta = float(h), float(w), float(l), float(x), float(y), float(z), float(theta)
            translation = np.linalg.inv(T) @ np.array([x,y,z,1]).T
            x, y, z, _ = translation
            theta = ry_to_rz(theta)
            label_list.append(np.array([x,y,z,h,w,l,theta]))
            # label = Label(kitti_object, float(h), float(w), float(l), float(x), float(y), float(z), float(theta))
            # label_list.append(label)
        if len(label_list)==0:
            return None
        return torch.from_numpy(np.vstack(label_list))
    
    def cal_target(self, gt_boxes):
        
        bev_unoriented_anchors = box2corner(self.anchors)
        bev_unoriented_gt = box2corner(gt_boxes)
        ious = bbox_overlaps(bev_unoriented_anchors, bev_unoriented_gt)

        _, max_id_gt_indices = ious.max(0)
        id_gts = torch.arange(len(max_id_gt_indices))
        max_anchor_ious, max_anchor_ious_indices = ious.max(1)
        mask = torch.where(max_anchor_ious<self.cfg['pos_threshold'])
        max_anchor_ious_indices[mask] = -1
        max_anchor_ious_indices[max_id_gt_indices] = id_gts
        val_indices = torch.where(max_anchor_ious_indices!=-1)
        pos_anchors = self.anchors[val_indices]
        pos_gt = gt_boxes[max_anchor_ious_indices[val_indices]]
        da = torch.sqrt(torch.pow(pos_anchors[:, 4], 2) + torch.pow(pos_anchors[:, 5], 2))
        ha = pos_anchors[:, 3]
        delta_x = (pos_gt[:,0] - pos_anchors[:,0])/da
        delta_y = (pos_gt[:,1] - pos_anchors[:,1])/da
        delta_z = (pos_gt[:,2] - pos_anchors[:,2])/ha
        delta_h = torch.log(pos_gt[:,3]/pos_anchors[:,3])
        delta_w = torch.log(pos_gt[:,4]/pos_anchors[:,4])
        delta_l = torch.log(pos_gt[:,5]/pos_anchors[:,5])
        delta_theta = pos_gt[:,6] - pos_anchors[:,6]
        u = torch.stack((delta_x, delta_y, delta_z, delta_l,delta_w, delta_h, delta_theta), dim=1).to(torch.float32)
        reg = torch.zeros((ious.shape[0], 7))
        reg[val_indices] = u
        reg = reg.reshape((*self.cfg['feature_size'], 14))
        pos = (max_anchor_ious_indices!=-1).to(torch.float32)
        pos = pos.reshape((*self.cfg['feature_size'], 2))
        neg = (max_anchor_ious<self.cfg['neg_threshold']).to(torch.float32)
        neg = neg.reshape((*self.cfg['feature_size'], 2))
        neg[torch.where((pos==1) & (neg==1))] = 0
        
        return pos, neg, reg
        
        
    def _parse_calib_file(self, calib_path):
        calib_data = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if ':' in line :
                    key, value = line.split(':', 1)
                    calib_data[key] = np.array([float(x) for x in value.split()])

    def get_sample(self, index):
        sample = {}
        image_path = self.image_list[index]
        label_path = self.label_list[index]
        pc_path = self.pc_list[index]
        calib_path = self.calib_list[index]

        calib = read_calib_file(calib_path)

        image = cv2.imread(image_path)
        

        labels = self._parse_label(label_path, calib_path)            
        if labels is None:
            return labels
        targets = self.cal_target(labels)
        point_cloud = read_point_cloud(pc_path)
        point_cloud = filter_point_cloud(self.cfg, point_cloud)
        tensor, voxels = point_cloud_to_tensor(self.cfg, point_cloud)
        voxels = torch.cat((torch.ones(len(voxels), 1, dtype=torch.int32), voxels), dim=-1)
        zero_pos = tensor.sum(-1)
        zero_pos = zero_pos==0

        return tensor, voxels, zero_pos, labels, calib, targets
        
    def __getitem__(self, index: Any) -> Any:
        item = self.get_sample(index)
        
        if item is None:
            for _ in range(self.cfg['max_fetch']):
                index = random.randint(0, self.__len__())
                item = self.get_sample(index)
                if item is not None:
                    return item
            
        return item
    
    

        

    
        