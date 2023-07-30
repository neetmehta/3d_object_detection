import numpy as np
import torch
import open3d as o3d
import yaml

def read_config(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    return data

def read_pc(file_path):
    pc = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return pc

def visualize_pc(pc):
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

