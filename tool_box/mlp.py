import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        net = []
        for i in range(len(layers)-2):
            net.append(nn.Linear(layers[i], layers[i+1]))
            net.append(nn.Tanh())
        net.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net)

        # Xavier 初始化
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class InversePINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.mlp = MLP(layers)
        # 学习 log_nu 以保证 nu>0；初值可按经验给个小粘度（大 Re）
        self.log_nu = nn.Parameter(torch.tensor([-4.6], dtype=torch.float64))  # nu ~ exp(-4) ≈ 0.0183 -> Re~54.6

    @property
    def nu(self):
        return torch.exp(self.log_nu)  # >0

    @property
    def Re(self):
        # 无量纲：Re = 1/nu
        return 1.0 / self.nu

    def forward(self, x):
        # x: [N,2] -> [u,v,p]
        return self.mlp(x)

