import time
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)

class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        net = []
        for i in range(len(layers)-2):
            net.append(nn.Linear(layers[i], layers[i+1]))
            net.append(nn.Tanh())
        net.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class LidCavity():
    def __init__(self,path,layers=[2, 64, 128, 256, 128, 64, 3],resol=(120,150)): 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.model = MLP(layers).to(self.device)
        self.model_path = path
        self.resol = resol
        self.gen_xy_meshgrid()

    def restore(self,path=None):
        path = self.model_path if path == None else path
        self.model.load_state_dict(torch.load(path))

    def gen_xy_meshgrid(self):
        x = np.linspace(0, 0.8, self.resol[0])
        y = np.linspace(0, 1, self.resol[1])
        self.XY_meshgrid = np.meshgrid(x, y)

    def prediction(self):
        start = time.time()
        self.model.eval()
        PLOT_X,PLOT_Y = self.resol
        X, Y = self.XY_meshgrid
        X = X.ravel()
        Y = Y.ravel()
        XY = np.stack([X, Y], axis=1).astype(np.float32)
        XY_t = torch.tensor(XY, device=self.device)
        out = self.model(XY_t).cpu().detach().numpy()
        U = out[:, 0].tolist()
        V = out[:, 1].tolist()
        print(f"Cost {time.time()-start}")
        return {
            'xmin':0,
            'xmax':0.8,
            'ymin':0,
            'ymax':1.0,
            'nx': PLOT_X,
            'ny': PLOT_Y,
            'u':U,
            'v':V
        }

    def prediction_3D(self):
        start = time.time()
        self.model.eval()
        PLOT_X,PLOT_Y = self.resol
        X, Y = self.XY_meshgrid
        X = X.ravel()
        Y = Y.ravel()
        Z = np.zeros_like(Y)
        Z[:] = 0.1
        XY = np.stack([X, Y, Z], axis=1).astype(np.float32)
        XY_t = torch.tensor(XY, device=self.device)
        out = self.model(XY_t).cpu().detach().numpy()
        U = out[:, 0].tolist()
        V = out[:, 1].tolist()
        print(f"Cost {time.time()-start}")
        return {
            'xmin':0,
            'xmax':0.8,
            'ymin':0,
            'ymax':1.0,
            'nx': PLOT_X,
            'ny': PLOT_Y,
            'u':U,
            'v':V
        }

def infer():
    layers = [2,16,32,64,64,32,16,3]
    tsonn = LidCavity(path="./model/model_gb0.6.pth",layers=layers)
    tsonn.restore()
    tsonn.prediction()

if __name__ == "__main__":
    infer()
