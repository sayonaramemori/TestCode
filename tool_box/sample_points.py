import numpy as np
import time
# import torch
import ctypes
import matplotlib.pyplot as plt
import platform

def get_current_os():
    os_name = platform.system()
    if os_name == "Windows":
        print("Running on Windows")
        return 0
    elif os_name == "Linux":
        print("Running on Linux")
        return 1
    elif os_name == "Darwin":
        print("Running on macOS")
        return 2
    else:
        print("Unknown OS:", os_name)
    # return os_name


def clamp(xy,eps,begin,end):
    return np.clip(xy, begin + eps, end - eps)                # clamped into [eps,1.0 - eps]


def sample_linear_2d(a, b, c, n_points=100, x_range=None, y_range=None,eps = 1e-6):
    """
    eps for () not []
    均匀采样二维线性函数 ax + by + c = 0 上的点。
    
    参数:
        a, b, c: float
            线性方程参数 ax + by + c = 0。
        n_points: int
            采样点数。
        x_range: tuple(float, float) or None
            x 的采样范围 (xmin, xmax)。若指定则按 x 均匀采样。
        y_range: tuple(float, float) or None
            y 的采样范围 (ymin, ymax)。若指定则按 y 均匀采样。
            
    返回:
        numpy.ndarray, shape = (n_points, 2)
            每一行为 [x, y]
    """
    if (x_range is None) and (y_range is None):
        raise ValueError("必须指定 x_range 或 y_range 之一。")

    if x_range is not None and y_range is not None:
        raise ValueError("只能指定一个范围（x_range 或 y_range）。")

    if b == 0 and y_range is None:
        raise ValueError("当 b=0 时，y 不可由 x 求出，必须指定 x_range。")

    if a == 0 and x_range is None:
        raise ValueError("当 a=0 时，x 不可由 y 求出，必须指定 y_range。")

    if x_range is not None:
        # xs = np.linspace(x_range[0], x_range[1], n_points)
        xs = np.random.rand(n_points) * (x_range[1] - x_range[0]) + x_range[0]
        xs = np.clip(xs,x_range[0]+eps,x_range[1]-eps)
        ys = -(a * xs + c) / b
    else:
        # ys = np.linspace(y_range[0], y_range[1], n_points)
        ys = np.random.rand(n_points) * (y_range[1] - y_range[0]) + y_range[0]
        ys = np.clip(ys,y_range[0]+eps,y_range[1]-eps)
        xs = -(b * ys + c) / a

    return np.stack([xs, ys], axis=1)

# For (0,1)
def sample_interior(n,dim=2,eps = 1e-6):
    # 均匀随机内点（不含边界）
    xy = np.random.rand(n, dim)                       # generate nx2 tensor
    xy = np.clip(xy, eps, 1.0 - eps)                # clamped into [eps,1.0 - eps]
    # return torch.tensor(xy, dtype=torch.float64, device=DEVICE)
    return xy

def filter_points(points, a, b, c, side='above'):
    """
    Remove points that are above or below the line a*x + b*y + c = 0.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2), where each row is [x, y].
    a, b, c : float
        Coefficients of the line a*x + b*y + c = 0.
    side : str
        'above' → remove points above the line
        'below' → remove points below the line

    Returns

-------
    np.ndarray
        Filtered array of points that remain.
    """
    # Compute signed distance (not normalized)
    values = a * points[:, 0] + b * points[:, 1] + c

    if side == 'above':
        mask = values < 0  # keep points below line
    elif side == 'below':
        mask = values > 0  # keep points above line
    else:
        raise ValueError("side must be 'above' or 'below'")

    return points[mask]

# xy should be numpy array
def get_vw_capi(xy,mp=True):
    t0 = time.time()
    size = len(xy)
    weights = np.zeros(size)
    libs = 0
    ost = get_current_os();
    if ost == 0:
        if mp:
            libs = ctypes.CDLL("ibvwmp.dll")
        else:
            libs = ctypes.CDLL("libvw.dll")
    else:
        if mp:
            libs = ctypes.CDLL("/home/kazusa/repo/PythonLearn-main/PINN/clib/libvwmp.so")
        else:
            libs = ctypes.CDLL("/home/l73/mybin/libvw.so")
    libs.vw2d_cal.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags="C_CONTIGUOUS"),
        ctypes.c_size_t
    ]
    libs.vw2d_cal(xy,weights,size)
    print(f"Calculate VW cost time = {time.time()-t0:.3f}s")
    return weights
    # return torch.tensor(weights, dtype=torch.float64, device=DEVICE)

def sample_uniform_interior(a,b,c,x_range=(0,1),c_range=1.0,eps = 1e-3):
    n_points = int((x_range[1] - x_range[0])/eps)
    x = np.linspace(x_range[0] + eps, x_range[1] - eps, n_points)
    def get_y(upon):
        y = (a*x + c)* (-1.0/b) + upon
        return np.stack([x, y], axis=1)
    res = get_y(eps)
    times = int(c_range/eps)
    for i in range(2,times):
        temp = get_y(i*eps)
        res = np.concatenate([res,temp])
    return res

def visual_points(points,path="points.png"):
    x = points[:, 0]
    y = points[:, 1]
    plt.scatter(x, y, color='blue', marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Points')
    plt.grid(True)
    plt.axis('equal')  # Keep aspect ratio square
    plt.show()
    # plt.savefig(path)
    plt.close()

def transform_2d_3d(xy,mat,bias=None):
    if bias is None:
        return xy @ mat
    else:
        return xy @ mat + bias

def sample_slant(n:int):
    xy = sample_interior(n).reshape(n,2)
    xy = transform_2d_3d(xy,np.array([
        [0.8,0],
        [0,1.0]
    ]))
    # xy = filter_points(xy,1,0,-0.8)
    xy = filter_points(xy,0.5,1,-0.15,side='below')
    return xy



if __name__ == "__main__":
    # xy = sample_interior(250)
    # w = get_vw_capi(xy,True)
    # get_vw_capi(xy,False)
    # xy = clamp(xy,1e-8,0,1)
    # xy = filter_points(xy,1,-1,0)
    # visual_points(xy)

    # xy = sample_uniform_interior(0,1,0,c_range=1,eps=6e-3)
    # print(xy.shape)
    # visual_points(xy)

    # xy = sample_uniform_interior(0.5,1,-0.15,c_range=1.25,eps=2e-2)
    """
    xy = sample_interior(100000)
    xy = filter_points(xy,1,0,-0.8)
    xy = filter_points(xy,0,1,0,side='below')
    xy = filter_points(xy,0.5,1,-0.15,side='below')
    xy = filter_points(xy,0,1,-1.0,side='above')
    distance = np.array([i[0]+i[1]**2 for i in xy])
    probability = distance/distance.sum()
    index = np.random.choice(a=len(distance),size=2000,replace=False,p=probability)
    visual_points(xy[index])
    """
    
    """
    left = sample_linear_2d(1,0,0,1000,None,(0,1))
    print(left)
    right = sample_linear_2d(1,0,-1,1000,None,(0,1))
    print(right)
    top = sample_linear_2d(0,1,-1,1000,(0,1),None)
    print(top)
    bot = sample_linear_2d(0,1,0,1000,(0,1),None)
    print(bot)
    x = np.concatenate([left,right,top,bot])
    visual_points(x,"left.png")
    """

    xy = sample_slant(10000)
    visual_points(xy,"left.png")


