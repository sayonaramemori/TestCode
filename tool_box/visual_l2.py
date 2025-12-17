import matplotlib.pyplot as plt
import numpy as np
import torch

@torch.no_grad()
def visualize_l2_error(
    model,
    xy_np: np.ndarray,
    uv_np: np.ndarray,
    filename: str,
    use_double: bool | None = None,
    batch_size: int = 65536,
    device: str | None = None,
):
    """
    可视化 PINN 预测与真值的 L2 误差。

    参数
    ----
    model      : PyTorch 模型；在 eval/no_grad 下调用，输入形状 (N,2)->(N,2或>=2)
    xy_np      : numpy.ndarray, 形状 (N,2)，分别为 (x,y)
    uv_np      : numpy.ndarray, 形状 (N,2)，分别为 (u,v) 真值
    filename   : str，保存图片的文件名（例如 'l2_error.png'）
    batch_size : int，分批推理的批大小，避免显存/内存溢出
    device     : 设备字符串，如 'cuda', 'cpu'；默认自动选择
    use_double : 若为 True 则使用 float64；默认与模型参数 dtype 对齐

    返回
    ----
    metrics : dict，包含
        - 'rms_l2'     : 全局 RMS L2 误差（对 (u,v) 的幅值，单位同数据）
        - 'rel_l2'     : 相对 L2 误差（对真值能量归一化）
        - 'per_point'  : 每个点的 L2 幅值误差数组，形状 (N,)
    """
    assert xy_np.ndim == 2 and xy_np.shape[1] == 2, "xy_np 需为 (N,2)"
    assert uv_np.ndim == 2 and uv_np.shape[1] == 2, "uv_np 需为 (N,2)"
    N = xy_np.shape[0]
    assert uv_np.shape[0] == N, "xy 与 uv 的点数不一致"

    model_was_training = model.training
    model.eval()

    # 设备与精度
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if use_double is None:
        # 与模型首个参数 dtype 对齐
        try:
            param_dtype = next(model.parameters()).dtype
        except StopIteration:
            param_dtype = torch.float32
        use_double = (param_dtype == torch.float64)

    dtype = torch.float64 if use_double else torch.float32

    # numpy -> torch
    xy_t = torch.from_numpy(xy_np).to(device=device, dtype=dtype)

    preds = []
    # 分批推理
    for i in range(0, N, batch_size):
        xb = xy_t[i:i + batch_size]
        yb = model(xb)
        # 取 (u,v) 前两维
        if isinstance(yb, (list, tuple)):
            yb = yb[0]
        yb = torch.as_tensor(yb)
        if yb.ndim == 1:
            yb = yb.unsqueeze(0)
        assert yb.shape[1] >= 2, "模型输出维度不足（至少应包含 u,v 两通道）"
        preds.append(yb[:, :2].detach())
    uv_pred_t = torch.cat(preds, dim=0)  # (N,2)
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

    # if model_was_training:
        # model.train()

    return {
        "rms_l2": rms_l2,
        "rel_l2": rel_l2,
        "per_point": err_np,
    }


def visualize_reference_uv(
    xy_np: np.ndarray,
    uv_np: np.ndarray,
    filename: str,
    scale: float = 1.0,
    density: int = 40,
    show_magnitude: bool = True,
):
    """
    可视化参考点 (x, y) 与其对应的速度场 (u, v)。

    参数
    ----
    xy_np : np.ndarray
        形状 (N, 2)，包含所有点的 (x, y) 坐标。
    uv_np : np.ndarray
        形状 (N, 2)，包含每个点的 (u, v) 分量。
    filename : str
        保存图片的文件名（例如 "reference_uv.png"）。
    scale : float, 默认 1.0
        控制箭头缩放因子，越大箭头越短。
    density : int, 默认 40
        若点数超过 density²，则进行下采样以避免图像过密。
    show_magnitude : bool, 默认 True
        若为 True，用颜色显示速度幅值；否则统一颜色。

    返回
    ----
    metrics : dict
        - "mean_speed" : 平均速度幅值
        - "max_speed"  : 最大速度幅值
    """
    assert xy_np.ndim == 2 and xy_np.shape[1] == 2, "xy_np 必须是 (N,2)"
    assert uv_np.ndim == 2 and uv_np.shape[1] == 2, "uv_np 必须是 (N,2)"
    N = xy_np.shape[0]
    assert uv_np.shape[0] == N, "xy 与 uv 数量不匹配"

    x, y = xy_np[:, 0], xy_np[:, 1]
    u, v = uv_np[:, 0], uv_np[:, 1]
    speed = np.sqrt(u**2 + v**2)

    # 计算统计量
    mean_speed = float(np.mean(speed))
    max_speed = float(np.max(speed))

    # 若点太多，则进行随机下采样以提升可视化效果
    if N > density * density:
        idx = np.random.choice(N, density * density, replace=False)
        x, y, u, v, speed = x[idx], y[idx], u[idx], v[idx], speed[idx]

    plt.figure(figsize=(7, 6))
    if show_magnitude:
        plt.quiver(
            x, y, u, v, speed, cmap="plasma", scale=scale * np.max(speed), width=0.0025
        )
        cb = plt.colorbar()
        cb.set_label("Velocity magnitude |u,v|")
    else:
        plt.quiver(x, y, u, v, color="tab:blue", scale=scale * np.max(speed), width=0.0025)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Reference Velocity Field — mean|u,v|={mean_speed:.3e}")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()

    return {"mean_speed": mean_speed, "max_speed": max_speed}


def visual_ref(xy_np, uv_np, pic_name="ref.png",cmap="viridis"):
    """
    可视化参考速度场 (u,v)，同时绘制速度幅值等值图与矢量箭头。

    参数
    ----
    xy_np : np.ndarray, 形状 (N,2)
        点坐标 (x,y)
    uv_np : np.ndarray, 形状 (N,2)
        速度分量 (u,v)
    pic_name : str
        输出图片文件名
    """
    X, Y = xy_np[:, 0], xy_np[:, 1]
    U, V = uv_np[:, 0], uv_np[:, 1]
    speed = np.sqrt(U**2 + V**2)

    plt.figure(figsize=(8, 10))
    plt.title("Velocity magnitude |u,v|")

    # 自动判断是否是规则网格
    x_unique = np.unique(X)
    y_unique = np.unique(Y)
    nx, ny = len(x_unique), len(y_unique)
    if nx * ny == X.shape[0]:  # 网格化数据
        # 重塑为网格
        Xg = X.reshape(ny, nx)
        Yg = Y.reshape(ny, nx)
        Ug = U.reshape(ny, nx)
        Vg = V.reshape(ny, nx)
        speed_g = speed.reshape(ny, nx)

        # 等值填色图 + 稀疏箭头场
        plt.contourf(Xg, Yg, speed_g, levels=30, cmap=cmap)
        plt.colorbar(label="|Velocity|")
        skip = (slice(None, None, max(1, ny // 25)), slice(None, None, max(1, nx // 25)))
        plt.quiver(Xg[skip], Yg[skip], Ug[skip], Vg[skip], color="k", scale=50)
    else:
        # 非规则散点，用 tricontourf
        tri = plt.tricontourf(X, Y, speed, levels=30, cmap=cmap)
        plt.colorbar(tri, label="|Velocity|")
        # 稀疏采样箭头
        step = max(1, X.shape[0] // 500)
        plt.quiver(X[::step], Y[::step], U[::step], V[::step], color="k", scale=50)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(pic_name, dpi=300)
    plt.close()


def visualize_p_l2_error(
    model,
    xy_np: np.ndarray,
    p_true_np: np.ndarray,
    filename: str = "p_l2_error.png",
    batch_size: int = 65536,
    device: str | None = None,
    use_double: bool | None = None,
    p_index: int | None = None,   # 若模型输出多通道，p所在通道索引；不传则自动猜测
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
    p_true_np = np.asarray(p_true_np).reshape(-1)
    assert p_true_np.shape[0] == xy_np.shape[0], "p_true_np 必须与 xy 点数一致"
    assert xy_np.ndim == 2 and xy_np.shape[1] == 2, "xy_np 必须是 (N,2)"
    N = xy_np.shape[0]

    was_training = model.training
    model.eval()

    # 设备与精度
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if use_double is None:
        try:
            param_dtype = next(model.parameters()).dtype
        except StopIteration:
            param_dtype = torch.float32
        use_double = (param_dtype == torch.float64)
    dtype = torch.float64 if use_double else torch.float32

    # numpy -> torch
    xy_t = torch.from_numpy(xy_np).to(device=device, dtype=dtype)

    # 推理并抽取压力通道
    preds = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = xy_t[i:i+batch_size]
            yb = model(xb)
            if isinstance(yb, (list, tuple)):
                yb = yb[0]
            yb = torch.as_tensor(yb, device=device, dtype=dtype)
            # 选择 p 通道
            if yb.ndim == 1:
                # 单通道，视为 p
                pb = yb
            else:
                C = yb.shape[-1]
                if C == 1:
                    pb = yb[:, 0]
                else:
                    # 自动猜测：优先用 p_index；未给则若 C>=3 用索引2，否则用最后一列
                    idx = p_index if p_index is not None else (2 if C >= 3 else C - 1)
                    pb = yb[:, idx]
            preds.append(pb.detach())
    p_pred_t = torch.cat(preds, dim=0)  # (N,)
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

    if was_training:
        model.train()

    return {
        "rms_l2": rms_l2,
        "rel_l2": rel_l2,
        "per_point": err_np,
    }

if __name__ == "__main__":
    data = np.loadtxt(f'../ref/slant-3000-3d-uniform.csv',delimiter=',')
    sliced_data = data[data[:,2] == 0.1]  # slice at z=0.1
    print(sliced_data.shape)
    xy_np = sliced_data[:,0:2]
    uv_np = sliced_data[:,3:5]
    # print(xy_np)
    visual_ref(xy_np, uv_np, pic_name="ref_slice-3000.png")
