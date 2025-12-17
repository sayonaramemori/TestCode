import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

# ===== 自动微分工具 =====
def gradients(y, x, order=1):
    """ compute dy/dx for scalar y and vector x; supports order=1 or 2 """
    if order == 1:
        return autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True)[0]
        # return autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True,allow_unused=True)[0]
    elif order == 2:
        g1 = gradients(y, x, order=1)
        # sum over components to get Laplacian contributions separately outside
        grads = []
        for i in range(g1.shape[-1]):
            gi = g1[..., i:i+1]
            g2 = autograd.grad(gi, x, grad_outputs=torch.ones_like(gi), create_graph=True, retain_graph=True)[0][..., i:i+1]
            grads.append(g2)
        return torch.cat(grads, dim=-1)  # diagonal second derivatives d2y/dx_i2
    else:
        raise NotImplementedError

# ===== PINN 损失构造 =====
def pde_residuals(model, xy, nu):
    xy.requires_grad_(True)
    out = model(xy)
    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]

    # 一阶导
    grads_u = gradients(u, xy, 1)  # [du/dx, du/dy]
    grads_v = gradients(v, xy, 1)  # [dv/dx, dv/dy]
    grads_p = gradients(p, xy, 1)  # [dp/dx, dp/dy]
    u_x, u_y = grads_u[:, 0:1], grads_u[:, 1:2]
    v_x, v_y = grads_v[:, 0:1], grads_v[:, 1:2]
    p_x, p_y = grads_p[:, 0:1], grads_p[:, 1:2]

    # 二阶导（拉普拉斯项）
    u_xx = gradients(u, xy, 2)[:, 0:1]
    u_yy = gradients(u, xy, 2)[:, 1:2]
    v_xx = gradients(v, xy, 2)[:, 0:1]
    v_yy = gradients(v, xy, 2)[:, 1:2]

    # 动量方程（稳态、无外力）: u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0
    #                             u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0
    f_u = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    f_v = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    # 连续性：u_x + v_y = 0
    f_c = u_x + v_y

    return f_u, f_v, f_c

