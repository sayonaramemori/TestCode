import matplotlib.pyplot as plt
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def streamplot(XY_meshgrid,model,pic_name="res.png",resol=(100,100),use_f64=False):
    X, Y = XY_meshgrid
    tt = np.float64 if use_f64 else np.float32
    XY = np.stack([X.ravel(), Y.ravel()], axis=1).astype(tt)
    XY_t = torch.tensor(XY, device=DEVICE)
    out = model(XY_t).cpu().numpy()
    PLOT_X,PLOT_Y = resol
    U = out[:, 0].reshape(PLOT_Y, PLOT_X)
    V = out[:, 1].reshape(PLOT_Y, PLOT_X)
    # P = out[:, 2].reshape(PLOT_N, PLOT_N)
    plt.figure(figsize=(8,10))
    plt.streamplot(X, Y, U, V, density=2)
    plt.title("streamplot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.axis("equal")
    # plt.show()
    plt.savefig(pic_name)
    plt.close()


@torch.no_grad()
def speed_mag(XY_meshgrid,model,pic_name="res.png",resol=(100,100),use_f64=False):
    PLOT_X,PLOT_Y = resol
    X, Y = XY_meshgrid
    ntype = np.float64 if use_f64 else np.float32
    XY = np.stack([X.ravel(), Y.ravel()], axis=1).astype(ntype)
    XY_t = torch.tensor(XY, device=DEVICE)
    out = model(XY_t).cpu().numpy()
    U = out[:, 0].reshape(PLOT_Y, PLOT_X)
    V = out[:, 1].reshape(PLOT_Y, PLOT_X)
    speed = np.sqrt(U**2 + V**2)
    plt.figure(figsize=(8,10))
    plt.title("Velocity magnitude |u|")
    plt.contourf(X, Y, speed, levels=30,cmap="viridis")
    plt.colorbar(label="|Velocity|")
    plt.quiver(X[::4,::4], Y[::4,::4], U[::4,::4], V[::4,::4], scale=50)
    plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
    plt.axis("equal")
    plt.savefig(pic_name)
    plt.close()

@torch.no_grad()
def pressure(XY_meshgrid,model,pic_name="pressure.png",resol=(100,100),use_f64=False):
    X, Y = XY_meshgrid
    ntype = np.float64 if use_f64 else np.float32
    XY = np.stack([X.ravel(), Y.ravel()], axis=1).astype(ntype)
    XY_t = torch.tensor(XY, device=DEVICE)
    out = model(XY_t).cpu().numpy()
    PLOT_X,PLOT_Y = resol
    P = out[:, 2].reshape(PLOT_Y, PLOT_X)
    plt.figure(figsize=(8,10))
    plt.title("Pressure p")
    plt.contourf(X, Y, P, levels=30)
    plt.colorbar()
    plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
    plt.axis("equal")
    plt.savefig(pic_name)
    plt.close()


@torch.no_grad()
def speed_l2_error(
    model,
    xy_np: np.ndarray,
    uv_np: np.ndarray,
    filename: str,
    use_f64: bool | None = None,
):
    """
    metrics : dict，包含
        - 'rms_l2'     : 全局 RMS L2 误差（对 (u,v) 的幅值，单位同数据）
        - 'rel_l2'     : 相对 L2 误差（对真值能量归一化）
        - 'per_point'  : 每个点的 L2 幅值误差数组，形状 (N,)
    """
    device = DEVICE
    dtype = torch.float64 if use_f64 else torch.float32
    # numpy -> torch
    xy_t = torch.from_numpy(xy_np).to(device=device, dtype=dtype)
    uv_pred_t = model(xy_t)[:,0:2]
    uv_true_t = torch.from_numpy(uv_np).to(device=device, dtype=dtype)

    # 每点 L2 幅值误差
    diff = uv_pred_t - uv_true_t
    per_point_err = torch.linalg.vector_norm(diff, ord=2, dim=1)  # (N,)

    # 全局 RMS L2（对幅值再做 RMS）
    rms_l2 = torch.sqrt(torch.mean(per_point_err**2)).item()

    # 相对 L2：||pred-true||_2 / ||true||_2（向量场能量归一化）
    num = torch.linalg.norm(diff)  # Frobenius over (N,2)
    den = torch.linalg.norm(uv_true_t)
    rel_l2 = (num / (den + 1e-30)).item()

    # 可视化：按误差着色的散点
    err_np = per_point_err.detach().cpu().numpy()
    x = xy_np[:, 0]
    y = xy_np[:, 1]

    plt.figure(figsize=(7, 6))
    sc = plt.scatter(x, y, c=err_np, s=8, cmap="viridis")
    cb = plt.colorbar(sc)
    cb.set_label("Per-point L2 error |Δ(u,v)|")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"L2 Error Map — RMS: {rms_l2:.3e} | Rel L2: {rel_l2:.3e}")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    return {
        "rms_l2": rms_l2,
        "rel_l2": rel_l2,
        "per_point": err_np,
    }

@torch.no_grad()
def pressure_l2_error(
    model,
    xy_np: np.ndarray,
    p_true_np: np.ndarray,
    filename: str = "p_l2_error.png",
    use_f64: bool = False,
):
    """
    可视化压力场的 L2 误差（逐点 |Δp| 等值图）并计算全局指标。

    参数
    ----
    model       : PyTorch 模型，输入 (N,2)->输出 (..., C)
    xy_np       : np.ndarray, 形状 (N,2)，坐标 (x,y)
    p_true_np   : np.ndarray, 形状 (N,)，参考压力 p_true
    filename    : str，保存图片文件名
    batch_size  : int，分批推理大小
    device      : str，'cuda' 或 'cpu'；默认自动
    use_double  : bool，是否使用 float64；默认与模型参数一致
    p_index     : int，若模型输出多通道时，压力所在通道的索引

    返回
    ----
    metrics : dict
        - 'rms_l2'    : 全局 RMS L2 误差
        - 'rel_l2'    : 相对 L2 误差（||pred-true||_2 / ||true||_2）
        - 'per_point' : 每点 |Δp| 数组 (N,)
    """
    N = xy_np.shape[0]
    dtype = torch.float64 if use_f64 else torch.float32
    device = DEVICE
    # numpy -> torch
    xy_t = torch.from_numpy(xy_np).to(device=device, dtype=dtype)

    # 推理并抽取压力通道
    p_pred_t = model(xy_t)[:,-1]
    p_true_t = torch.from_numpy(p_true_np).to(device=device, dtype=dtype)

    # 误差与指标
    diff = p_pred_t - p_true_t
    per_point_err = torch.abs(diff)                # 每点 |Δp|
    rms_l2 = torch.sqrt(torch.mean(diff**2)).item()
    rel_l2 = (torch.linalg.norm(diff) / (torch.linalg.norm(p_true_t) + 1e-30)).item()

    # 可视化：|Δp| 等值图（网格/散点自适应）
    err_np = per_point_err.detach().cpu().numpy()
    X = xy_np[:, 0]
    Y = xy_np[:, 1]

    plt.figure(figsize=(7, 6))
    plt.title(f"Pressure L2 Error |Δp| — RMS: {rms_l2:.3e} | Rel: {rel_l2:.3e}")

    x_unique = np.unique(X)
    y_unique = np.unique(Y)
    nx, ny = len(x_unique), len(y_unique)

    if nx * ny == N:
        # 规则网格
        Xg = X.reshape(ny, nx)
        Yg = Y.reshape(ny, nx)
        Eg = err_np.reshape(ny, nx)
        cs = plt.contourf(Xg, Yg, Eg, levels=30, cmap="magma")
        cb = plt.colorbar(cs)
        cb.set_label("|Δp|")
    else:
        # 非规则点
        cs = plt.tricontourf(X, Y, err_np, levels=30, cmap="magma")
        cb = plt.colorbar(cs)
        cb.set_label("|Δp|")

    plt.xlabel("x"); plt.ylabel("y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "rms_l2": rms_l2,
        "rel_l2": rel_l2,
        "per_point": err_np,
    }


