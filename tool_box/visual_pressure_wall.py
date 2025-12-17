import numpy as np
import matplotlib.pyplot as plt
import torch

@torch.no_grad()
def visual_pwall(model,XY,p_ref,output="pressure_on_wall.png",use_double=False):
    tt = torch.float64 if use_double else torch.float32
    nt = np.float64 if use_double else np.float32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    XY_t = torch.tensor(XY, dtype=tt,device=DEVICE)
    out = model(XY_t).cpu().numpy().astype(nt)
    p = out[:,-1]
    index = np.arange(p_ref.shape[0])
    plt.figure(figsize=(8, 6))
    plt.plot(index,p - p.mean(),label="Prediction")
    plt.plot(index,p_ref - p_ref.mean(),label="Ref")
    plt.title(f'Relative Pressure')
    plt.xlabel('height')
    plt.ylabel('P relative')
    plt.grid()
    plt.legend()
    plt.savefig(output)
    plt.close()



