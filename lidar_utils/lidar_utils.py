import numpy as np
import torch
import open3d as o3d
import yaml

def read_config(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    return data

def read_point_cloud(file_path):
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return pc

def visualize_pc(pc, bb=None):
    
    points, intensities = pc[:, :3], pc[:, 3]
    color = np.array([1.0, 1.0, 1.0]).astype(np.float64)
    color = intensities[:, None] @ color[None, :]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(1 - color)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud with Intensities', width=800, height=600)
    vis.get_render_option().background_color = np.asarray([0, 0, 0])  # Set the background color to black
    vis.add_geometry(pcd)
    if bb is not None:
        bounding_boxes = []
        lines = [[0, 1], [1, 2], [2, 3], [0, 3],
                [4, 5], [5, 6], [6, 7], [4, 7],
                [0, 4], [1, 5], [2, 6], [3, 7]]

        # Use the same color for all lines
        colors = [[0, 1, 0] for _ in range(len(lines))]
        for i in bb:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(i)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(line_set)
    vis.run()
    vis.destroy_window()

def filter_point_cloud(cfg, pc):

    x_min, x_max = cfg['x_min'], cfg['x_max']
    y_min, y_max = cfg['y_min'], cfg['y_max']
    z_min, z_max = cfg['z_min'], cfg['z_max']

    x_filter = np.logical_and(pc[:, 0]>=x_min, pc[:, 0]<=x_max)
    y_filter = np.logical_and(pc[:, 1]>=y_min, pc[:, 1]<=y_max)
    z_filter = np.logical_and(pc[:, 2]>=z_min, pc[:, 2]<=z_max)
    xy_filter = np.logical_and(x_filter, y_filter)
    xyz_filter = np.logical_and(xy_filter, z_filter)
    pc = pc[xyz_filter]
    return pc

def voxelize(point_cloud, cfg):

    x_min = cfg['x_min'] 
    y_min = cfg['y_min'] 
    z_min = cfg['z_min']
    vd, vh, vw = cfg['vd'], cfg['vh'], cfg['vw']


    points, _ = point_cloud[:, :3], point_cloud[:, 3]
    voxel_coords = ((points - np.array([x_min, y_min, z_min]))/np.array([vw, vh, vd])).astype(np.int64)
    return voxel_coords

def point_cloud_to_tensor(cfg, point_cloud):
    voxel_coords = voxelize(point_cloud, cfg)
    T = cfg['T']
    unique_voxel_coords, inverse_index = np.unique(voxel_coords, axis=0, return_inverse=True)
    K = unique_voxel_coords.shape[0]
    tensor = np.zeros((K, T, 7))
    for i in range(len(unique_voxel_coords)):
        indices = np.flatnonzero(inverse_index==i)
        # print(unique_voxel_coords[i])
        row = point_cloud[indices][:T, :]
        aug = row[:,:3] - row[:,:3].mean(0)
        new_row = np.hstack((row, aug))
        tensor[i, :min(T, len(indices)), :] = new_row
        
    return torch.from_numpy(tensor), torch.from_numpy(unique_voxel_coords)

def read_calib_file(file_path):
    
    with open(file_path, 'r') as f:
        calib_lines = f.readlines()
        
    calib = {}
    calib['cam_intrinsics'] = np.array([float(i) for i in calib_lines[2].split()[1:]]).reshape(3,4)
    calib['R0_rect'] = np.array([float(i) for i in calib_lines[2].split()[1:]]).reshape(3,4)
    calib['Tr_velo_to_cam'] = np.array([float(i) for i in calib_lines[5].split()[1:]]).reshape(3,4)
    return calib

def read_label(file_path):
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    label_list = []
    for label in lines:
        class_name, _, _, _, _, _, _, _, l, b, h, x, y, z, theta = label.split()
        if class_name!='DontCare':
            label_list.append([class_name, float(l), float(b), float(h), float(x), float(y), float(z), float(theta)])
        
    return label_list

def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])
    

def label_2_bb(label, tr_velo_2_cam):
    _, h, w, l, x, y, z, theta = label
    T = np.zeros((4,4))
    T[:3,:] = tr_velo_2_cam
    T[3,3] = 1
    translation = np.linalg.inv(T) @ np.array([x,y,z,1]).T
    x, y, z, _ = translation
    
    # R = np.array([
    #     [np.cos(theta), -np.sin(theta), 0.0],
    #     [np.sin(theta), np.cos(theta), 0.0],
    #     [0.0, 0.0, 1.0]])

    theta = -theta + np.pi/2
    R = rotz(theta)
    
    bounding_box = np.array([
        [-l/2, -l/2, l/2, l/2, -l/2, -l/2, l/2, l/2],
        [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        [0, 0, 0, 0, h, h, h, h]])
    corners_3d = np.dot(R, bounding_box)
    corners_3d[0,:] = corners_3d[0,:] + x
    corners_3d[1,:] = corners_3d[1,:] + y
    corners_3d[2,:] = corners_3d[2,:] + z

    return np.transpose(corners_3d)
        
def visualize_bb(point_cloud, label, calib):
    bb = []
    for i in label:
        bb.append(label_2_bb(i, calib['Tr_velo_to_cam']))
        
    visualize_pc(point_cloud, bb)
    
def visualize_bb_from_file(velo_file, label_file, calib_file, config_file):
    point_cloud = read_point_cloud(velo_file)
    cfg = read_config(config_file)
    point_cloud = filter_point_cloud(cfg, point_cloud)
    calib = read_calib_file(calib_file)
    label = read_label(label_file)

    visualize_bb(point_cloud, label, calib)