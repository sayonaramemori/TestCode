import matplotlib.pyplot as plt
import numpy as np
import torch

def visual_pinn_stream(model,pic_name="stream.png",resol=100,use_f64=False):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PLOT_N = resol
    with torch.no_grad():
        xx = np.linspace(0, 1, resol)
        yy = np.linspace(0, 1, resol)
        X, Y = np.meshgrid(xx, yy)
        tt = np.float64 if use_f64 else np.float32
        XY = np.stack([X.ravel(), Y.ravel()], axis=1).astype(tt)
        XY_t = torch.tensor(XY, device=DEVICE)
        out = model(XY_t).cpu().numpy()
        U = out[:, 0].reshape(PLOT_N, PLOT_N)
        V = out[:, 1].reshape(PLOT_N, PLOT_N)
        # P = out[:, 2].reshape(PLOT_N, PLOT_N)

        plt.figure(figsize=(8,10))
        plt.streamplot(X, Y, U, V, density=1.8)
        plt.title("Lid-Driven Cavity")
        plt.xlabel("x")
        plt.ylabel("y")
        # plt.show()
        plt.savefig(pic_name)
        plt.close()


@torch.no_grad()
def visual_pinn_result(model,pic_name="res.png",resol=100,use_f64=False):
    PLOT_N = resol
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = np.linspace(0, 1, PLOT_N)
    y = np.linspace(0, 1, PLOT_N)
    X, Y = np.meshgrid(x, y)
    ntype = np.float64 if use_f64 else np.float32
    XY = np.stack([X.ravel(), Y.ravel()], axis=1).astype(ntype)
    XY_t = torch.tensor(XY, device=DEVICE)
    out = model(XY_t).cpu().numpy()
    U = out[:, 0].reshape(PLOT_N, PLOT_N)
    V = out[:, 1].reshape(PLOT_N, PLOT_N)
    # P = out[:, 2].reshape(PLOT_N, PLOT_N)

    speed = np.sqrt(U**2 + V**2)

    plt.figure(figsize=(8,10))
    plt.title("Velocity magnitude |u|")
    plt.contourf(X, Y, speed, levels=30)
    plt.colorbar()
    plt.quiver(X[::4,::4], Y[::4,::4], U[::4,::4], V[::4,::4], scale=50)
    plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
    plt.savefig(pic_name)
    plt.close()

@torch.no_grad()
def visual_pinn_pressure(model,pic_name="stream.png",resol=100,use_f64=False):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PLOT_N = resol
    # with torch.no_grad():
    model.eval()
    xx = np.linspace(0, 1, resol)
    yy = np.linspace(0, 1, resol)
    X, Y = np.meshgrid(xx, yy)
    ntype = np.float64 if use_f64 else np.float32
    XY = np.stack([X.ravel(), Y.ravel()], axis=1).astype(ntype)
    XY_t = torch.tensor(XY, device=DEVICE)
    out = model(XY_t).cpu().numpy()
    P = out[:, 2].reshape(PLOT_N, PLOT_N)

    plt.figure(figsize=(8,10))
    plt.title("Pressure p")
    plt.contourf(X, Y, P, levels=30)
    plt.colorbar()
    plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
    plt.savefig(pic_name)
    plt.close()

@torch.no_grad()
def visual_pinn_result_slant(model,pic_name="res.png",resol=100,use_f64=False):
    PLOT_N = resol
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = np.linspace(0, 0.8, PLOT_N)
    y = np.linspace(0, 1, PLOT_N)
    X, Y = np.meshgrid(x, y)
    for i in range(len(X)):
        for j in range(len(X[i])):
            temp = 0.5 * X[i][j] + Y[i][j] - 0.15
            if temp<0:
                Y[i][j] = 0.15 - 0.5*X[i][j]
    ntype = np.float64 if use_f64 else np.float32
    XY = np.stack([X.ravel(), Y.ravel()], axis=1).astype(ntype)
    XY_t = torch.tensor(XY, device=DEVICE)
    out = model(XY_t).cpu().numpy()
    U = out[:, 0].reshape(PLOT_N, PLOT_N)
    V = out[:, 1].reshape(PLOT_N, PLOT_N)
    # P = out[:, 2].reshape(PLOT_N, PLOT_N)

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
def visual_pinn_pressure_slant(model,pic_name="stream.png",resol=100,use_f64 = False):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PLOT_N = resol
    # with torch.no_grad():
    model.eval()
    xx = np.linspace(0, 0.8, resol)
    yy = np.linspace(0, 1, resol)
    X, Y = np.meshgrid(xx, yy)
    for i in range(len(X)):
        for j in range(len(X[i])):
            temp = 0.5 * X[i][j] + Y[i][j] - 0.15
            if temp<0:
                Y[i][j] = 0.15 - 0.5*X[i][j]
    ntype = np.float64 if use_f64 else np.float32
    XY = np.stack([X.ravel(), Y.ravel()], axis=1).astype(ntype)
    XY_t = torch.tensor(XY, device=DEVICE)
    out = model(XY_t).cpu().numpy()
    P = out[:, 2].reshape(PLOT_N, PLOT_N)

    plt.figure(figsize=(8,10))
    plt.title("Pressure p")
    plt.contourf(X, Y, P, levels=30)
    plt.colorbar()
    plt.xlabel("x"); plt.ylabel("y"); plt.tight_layout()
    plt.axis("equal")
    plt.savefig(pic_name)
    plt.close()


@torch.no_grad()
def visual_pinn_stream_slant(model,pic_name="stream.png",resol=100,use_f64=False):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PLOT_N = resol
    xx = np.linspace(0, 0.8, resol)
    yy = np.linspace(0, 1, resol)
    X, Y = np.meshgrid(xx, yy)
    ntype = np.float64 if use_f64 else np.float32
    XY = np.stack([X.ravel(), Y.ravel()], axis=1).astype(ntype)
    XY_t = torch.tensor(XY, device=DEVICE)
    out = model(XY_t).cpu().numpy()
    U = out[:, 0]
    V = out[:, 1]
    for i in range(XY_t.__len__()):
        temp = 0.5 * XY_t[i][0] + XY_t[i][1] - 0.15
        if temp<0:
            U[i] = 0
            V[i] = 0
    # P = out[:, 2].reshape(PLOT_N, PLOT_N)
    U = out[:, 0].reshape(PLOT_N, PLOT_N)
    V = out[:, 1].reshape(PLOT_N, PLOT_N)

    plt.figure(figsize=(8,10))
    plt.streamplot(X, Y, U, V, density=1.8)
    plt.title("Lid-Driven Cavity")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.axis("equal")
    # plt.show()
    plt.savefig(pic_name)
    plt.close()

