import torch
import numpy as np
import matplotlib.pyplot as plt
# from sample_points import filter_points

def show_samples_3d(points, title="3D sample points", max_points=200000):
    """
    可视化三维采样点分布（仅散点图）
    points: torch.Tensor 或 np.ndarray，形状 (N,3)
    max_points: 限制显示的点数（太多点绘图会很慢）
    """
    # 转为 numpy
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    assert points.shape[1] == 3, "points 必须是 (N,3) 形状"

    # 下采样
    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[idx]

    x, y, z = points[:,0], points[:,1], points[:,2]

    # 绘图
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=2)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def sample_interior(n,dim=2,eps = 1e-6):
    # 均匀随机内点（不含边界）
    xy = np.random.rand(n, dim)                       # generate nx2 tensor
    xy = np.clip(xy, eps, 1.0 - eps)                # clamped into [eps,1.0 - eps]
    # return torch.tensor(xy, dtype=torch.float64, device=DEVICE)
    return xy

def sample_slant(n:int):
    xy = sample_interior(n)
    xy = filter_points(xy,1,0,-0.8)
    xy = filter_points(xy,0.5,1,-0.15,side='below')
    return xy

# xy is Nx2 and mat is 2x3
def transform_2d_3d(xy,mat,bias=None):
    if bias is None:
        return xy @ mat
    else:
        return xy @ mat + bias

def test_boundary(n:int = 50000):
    top = transform_2d_3d(sample_interior(int(n*0.07)),np.array([
        [0.8,0,0],
        [0,0,0.2]
    ]),np.array([0,1,0]))
    bot = transform_2d_3d(sample_interior(int(n*0.05)),np.array([
        [0.5,0,0],
        [0,0,0.2]
    ]),np.array([0.3,0,0]))
    right = transform_2d_3d(sample_interior(int(n*0.08)),np.array([
        [0,1,0],
        [0,0,0.2]
    ]),np.array([0.8,0,0]))
    left = transform_2d_3d(sample_interior(int(n*0.09)),np.array([
        [0,0.85,0],
        [0,0,0.2]
    ]),np.array([0,0.15,0]))
    slant = transform_2d_3d(sample_interior(int(n*0.03)),np.array([
        [0.3,-0.15,0],
        [0,0,0.2]
    ]),np.array([0,0.15,0]))
    front= transform_2d_3d(sample_slant(int(0.35*n)),np.array([
        [1,0,0],
        [0,1,0]
    ]),np.array([0,0,0.2]))
    back= transform_2d_3d(sample_slant(int(0.35*n)),np.array([
        [1,0,0],
        [0,1,0]
    ]))
    xyz = np.concatenate([top,bot,right,left,slant,front,back])
    return xyz


def test_interior(n:int = 120000):
    xyz_slant = transform_2d_3d(sample_interior(int(n*0.05),dim=3),np.array([
        [0.3,0,0],
        [0,0.15,0],
        [0,0,0.2]
    ]))
    idx = []
    for i in range(xyz_slant.__len__()):
        y_temp = -0.5*xyz_slant[i][0] + 0.15
        if y_temp - xyz_slant[i][1] >= 1e-6:
            continue
        idx.append(i)
    xyz_slant = xyz_slant[idx]
    xyz_left = transform_2d_3d(sample_interior(int(n*0.32),dim=3),np.array([
        [0.3,0,0],
        [0,0.85,0],
        [0,0,0.2]
    ]),np.array([0,0.15,0]))
    xyz_right = transform_2d_3d(sample_interior(int(n*0.63),dim=3),np.array([
        [0.5,0,0],
        [0,1,0],
        [0,0,0.2]
    ]),np.array([0.3,0,0]))
    xyz = np.concatenate([xyz_slant,xyz_left,xyz_right])
    return xyz
    # show_samples_3d(xyz)


if __name__ == "__main__":
    xyz_i = test_interior()
    xyz_bc = test_boundary()
    xyz = np.concatenate([xyz_i,xyz_bc])
    show_samples_3d(xyz)


