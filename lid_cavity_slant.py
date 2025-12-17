import time
import numpy as np
import torch
from tool_box import *
from tool_box import visual_2d
from read_toml import *

cfg = get_cfg('settings')
use_f64 = cfg.use_f64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
ntype = np.float64 if use_f64 else np.float32
ttype = torch.float64 if use_f64 else torch.float32

torch.set_default_dtype(ttype)

def fwd_gradients(Y, x):
    dummy = torch.ones_like(Y)
    G = torch.autograd.grad(Y, x, dummy, create_graph= True)[0]
    return G

if cfg.seed != None:
    print("Set Seed for Torch")
    torch.manual_seed(cfg.seed)
    # np.random.seed(SEED)

class LidCavity():
    def __init__(self,output_dir,nb,ns,nv,Re,ref='./points_4000.csv',layers=[2, 64, 128, 256, 128, 64, 3],dtau = None,weight_bc = 4.0,weight_pde = 1.0,gamma = 0.33,gamma_bound=1.0):
        self.gamma_bound = gamma_bound
        self.gamma = gamma
        self.target_dir = output_dir
        self.weight_bc = weight_bc
        self.weight_pde = weight_pde
        self.loss = self.loss_pde = self.loss_bc = torch.tensor([0.0])
        self.Re = Re
        self.model_path = f'./model/model_gb{gamma_bound}.pth'
        self.loss_target = f'./loss_record/loss{gamma_bound}.csv'
        self.nb = nb
        self.ns = ns
        self.nv = nv
        self.iter = 500
        self.ns_base = ns
        self.loss_cmp = 0.99
        self.l2_min = 0.99
        self.dtau = 0.5 if dtau == None else dtau
        self.device = DEVICE
        self.model = MLP(layers).to(self.device)
        self.sample_slant(ns)
        self.sample_boundary(nb)
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=500,
            history_size=100,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=0
        )
        points = np.loadtxt(ref,delimiter=',')
        filter_points = points[points[:,2] == 0]
        self.wall_xy_p = filter_points[filter_points[:,0]==0.8]
        self.wall_xy = self.wall_xy_p[:,0:2]
        self.wall_p = self.wall_xy_p[:,-1]
        self.xy_ref = filter_points[:,0:2]
        self.uv_ref = filter_points[:,3:5]
        self.p_ref = filter_points[:,-1]
        ref_V_mag = np.linalg.norm(self.uv_ref,axis=1).reshape(-1,1)

        self.lossR = LossRecorder(col=['loss','pde_loss','bc_loss','L2','cost'])
        
        self.X_ref = torch.from_numpy(self.xy_ref.astype(ntype)).to(DEVICE)
        self.V_ref = torch.from_numpy(ref_V_mag.astype(ntype)).to(DEVICE)
        self.save()
        self.gen_xy_meshgrid()

    def gen_xy_meshgrid(self):
        self.resol = (120,150)
        x = np.linspace(0, 0.8, self.resol[0])
        y = np.linspace(0, 1, self.resol[1])
        X, Y = np.meshgrid(x, y)
        for i in range(len(X)):
            for j in range(len(X[i])):
                temp = 0.5 * X[i][j] + Y[i][j] - 0.15
                if temp < 0:
                    Y[i][j] = 0.15 - 0.5*X[i][j]
        self.XY_meshgrid = (X,Y)
        self.XY_meshgrid_origin = np.meshgrid(x, y)
 
    def save(self):
        torch.save(self.model.state_dict(),self.model_path)

    def restore(self,path=None):
        # self.model = torch.load(path,weights_only=False)
        path = self.model_path if path == None else path
        self.model.load_state_dict(torch.load(path))

    def sample_slant(self,n:int):
        xy = sample_slant(n)
        self.xy_f = torch.tensor(xy, dtype=ttype, device=DEVICE)

    def sample_boundary(self,n:int):
        get_n = lambda f : int(n*f)
        top = sample_linear_2d(0,1,-1,get_n(0.23),(0,0.8),None)
        left = sample_linear_2d(1,0,0,get_n(0.24),None,(0.15,1))
        right = sample_linear_2d(1,0,-0.8,get_n(0.28),None,(0,1))
        bottom = sample_linear_2d(0,1,0,get_n(0.14),(0.3,0.8),None)
        slant = sample_linear_2d(0.5,1,-0.15,get_n(0.12),(0,0.3),None)

        self.top = torch.tensor(top,dtype=ttype,device=DEVICE)
        self.left = torch.tensor(left,dtype=ttype,device=DEVICE)
        self.right = torch.tensor(right,dtype=ttype,device=DEVICE)
        self.bottom = torch.tensor(bottom,dtype=ttype,device=DEVICE)
        self.slant= torch.tensor(slant,dtype=ttype,device=DEVICE)
        self.boundary = torch.cat([self.top,self.bottom,self.left,self.right,self.slant])

        begin = 0; end = begin + top.__len__();self.idx_top = slice(begin,end)
        begin = end; end = begin + left.__len__();self.idx_left = slice(begin,end)
        begin = end; end = begin + right.__len__();self.idx_right = slice(begin,end)
        begin = end; end = begin + bottom.__len__();self.idx_bot = slice(begin,end)
        begin = end; end = begin + slant.__len__();self.idx_slant = slice(begin,end)

    def TimeStepping(self):
        X = self.xy_f
        pred = self.model(X)
        u = pred[:,0:1];  v = pred[:,1:2];   p = pred[:,2:3];
        self.U0 = torch.cat([u,v,p]).detach()
       
    def pde_residuals(self):
        X = self.xy_f
        X.requires_grad_(True)
        pred = self.model(X)
        u = pred[:,0:1];  v = pred[:,1:2];   p = pred[:,2:3];
              
        u_xy = fwd_gradients(u, X)
        v_xy = fwd_gradients(v, X)
        p_xy = fwd_gradients(p, X)
        u_x = u_xy[:,0:1]; u_y = u_xy[:,1:2]
        v_x = v_xy[:,0:1]; v_y = v_xy[:,1:2]
        p_x = p_xy[:,0:1]; p_y = p_xy[:,1:2]
        u_xx = fwd_gradients(u_x, X)[:,0:1]
        u_yy = fwd_gradients(u_y, X)[:,1:2]
        v_xx = fwd_gradients(v_x, X)[:,0:1]
        v_yy = fwd_gradients(v_y, X)[:,1:2]
        res_u = u*u_x + v*u_y + p_x - 1.0/self.Re*(u_xx + u_yy)
        res_v = u*v_x + v*v_y + p_y - 1.0/self.Re*(v_xx + v_yy)
        res_rho = u_x + v_y
        U1 = torch.cat([u,v,p])
        R1 = torch.cat([res_u,res_v,res_rho])
        
        # Residuals == - N(u) | here is Residuals
        msef = 1/self.dtau**2*((U1 - self.U0 + self.dtau*R1)**2).mean()
        return msef

    # Calculate the L2 Error
    @torch.no_grad()
    def cal_L2(self):
        self.model.eval()
        pred = self.model(self.X_ref)
        u = pred[:,0:1]; v = pred[:,1:2];
        V = torch.sqrt(u**2 + v**2).detach().reshape(-1,1)
        mses = torch.norm((V - self.V_ref).reshape(-1),2) / torch.norm((self.V_ref).reshape(-1),2)
        return mses

    def bc_residuals_soft(self):
        with torch.set_grad_enabled(True):
            self.boundary.requires_grad_(True)
            out = self.model(self.boundary)
            u = out[:, 0:1]
            v = out[:, 1:2]

            bc_top_u = u[self.idx_top] - 0.0
            bc_top_v = v[self.idx_top] - 0.0
            bc_bot_u = u[self.idx_bot] - 0.0
            bc_bot_v = v[self.idx_bot] - 0.0
            bc_left_u = u[self.idx_left] - 0.0
            bc_left_v = v[self.idx_left] - 0.0
            bc_right_u = u[self.idx_right] - 0.0
            bc_right_v = v[self.idx_right] - 0.0
            bc_slant_u = u[self.idx_slant] - .89; bc_slant_v = v[self.idx_slant] + .445
            bc_u = torch.cat([bc_top_u, bc_bot_u, bc_left_u, bc_right_u, bc_slant_u], dim=0)
            bc_v = torch.cat([bc_top_v, bc_bot_v, bc_left_v, bc_right_v, bc_slant_v], dim=0)

            return bc_u, bc_v

    def train_adam_original(self,epoch):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        for ep in range(1, epoch):
            self.model.train()
            optimizer.zero_grad()
            f_u, f_v, f_c = pde_residuals(self.model,self.xy_f,1.0/self.Re)
            bc_u, bc_v = self.bc_residuals_soft()
            # p_anchor_loss = self.p_anchor_residual().pow(2).mean()
            loss_bc = (bc_u.pow(2).mean() + bc_v.pow(2).mean())
            loss_pde = (f_u.pow(2).mean() + f_v.pow(2).mean() + 2.0 * f_c.pow(2).mean())

            # loss = loss_pde + loss_bc + p_anchor_loss
            loss = loss_pde + 4.0*loss_bc
            loss.backward()
            optimizer.step()
            scheduler.step()
            l2 = self.cal_L2()
            if ep%100==0 and ep>1:
                print(f"[Adam] Epoch {ep:5d} | "
                      f"loss={loss.item():.4e} | pde={loss_pde.item():.4e} | bc={loss_bc.item():.4e} | l2={l2:.4e}")
                torch.save(self.model.state_dict(),self.model_path)
        self.visualize();

    def train_adam(self,epoch=100,iter_n=3):
        for _ in range(iter_n):
            optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
            self.TimeStepping()
            for ep in range(1, epoch):
                self.model.train()
                optimizer.zero_grad()
                loss_pde = self.pde_residuals()
                bc_u, bc_v = self.bc_residuals_soft()
                loss_bc = (bc_u.pow(2).mean() + bc_v.pow(2).mean())

                loss = self.weight_pde * loss_pde + self.weight_bc * loss_bc
                loss.backward()
                optimizer.step()
                l2 = self.cal_L2()
                if ep%50 == 0:
                    print(f"[Adam] Epoch {ep:5d} | loss={loss.item():.4e} | pde={loss_pde.item():.4e} | bc={loss_bc.item():.4e} | l2={l2:.4e}")
        self.visualize()
        self.save()

    def decontructor_hook(self):
        self.lossR.save_as_csv(f"./loss_record/loss-slant-re{self.Re}.csv")
        self.lossR.plot(f'{self.target_dir}/l2_loss.png')
        self.visualize_best()

    def visualize_best(self,path = None):
        self.model.load_state_dict(torch.load(self.model_path if path == None else path))
        self.model.eval()
        self.visualize(index='best')
        visual_ref(self.xy_ref,self.uv_ref,f"{self.target_dir}/ref_uv.png")

    def visualize(self,index=None):
        target = f"{self.target_dir}/{index}"
        os.makedirs(target, exist_ok=True)
        visual_2d.speed_mag(self.XY_meshgrid,self.model,f"{target}/u.png",resol=self.resol,use_f64=use_f64)
        visual_2d.pressure(self.XY_meshgrid,self.model,f"{target}/p.png",resol=self.resol,use_f64=use_f64)
        visual_2d.streamplot(self.XY_meshgrid_origin,self.model,f"{target}/s.png",resol=self.resol,use_f64=use_f64)
        visual_2d.speed_l2_error(self.model,self.xy_ref,self.uv_ref,f"{target}/u_l2.png",use_f64=use_f64)
        visual_2d.pressure_l2_error(self.model,self.xy_ref,self.p_ref,f"{target}/p_l2.png",use_f64=use_f64)
        visual_pwall(self.model,self.wall_xy,self.wall_p,f"{target}/pwall.png",use_double=use_f64)

    def train(self,epoch=300):
        LBFGS_step = 1
        best_epoch = 0
        for i in range(epoch):
            t0 = time.time()
            if self.gamma<0.75: 
                self.gamma += 0.005
            def closure_for_lbfgs():
                nonlocal LBFGS_step
                self.model.train()
                self.optimizer.zero_grad()
                self.loss_pde = self.pde_residuals()
                bc_u, bc_v = self.bc_residuals_soft()
                self.loss_bc = (bc_u.pow(2).mean() + bc_v.pow(2).mean())
                gamma = self.gamma * self.loss_pde.item()/(self.loss_pde.item()+self.loss_bc.item())
                self.weight_pde = 2.0 * gamma;
                self.weight_bc = 2.0 - self.weight_pde
                self.true_gamma = self.weight_pde/self.weight_bc
                self.loss = self.weight_pde * self.loss_pde + self.weight_bc * self.loss_bc
                self.loss.backward()
                LBFGS_step += 1
                return self.loss
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), max_iter=self.iter)
            self.sample_slant(self.ns)
            self.sample_boundary(self.nb)
            self.TimeStepping()
            self.optimizer.step(closure_for_lbfgs)

            cost = int(time.time()-t0)
            l2 = self.cal_L2(); better = self.l2_min > l2
            nan_flag = not torch.isfinite(self.loss); lack_progress_flag = cost <= 1; error_flag = self.loss.item() > 4.5 * self.loss_cmp
            print(f"Epoch {i:3d}| ns {self.ns} |Re {self.Re}| L2 {l2:.4e} | PDE {self.loss_pde.item():.4e} | BC {self.loss_bc.item():.4e}| Loss {self.loss.item():.4e} | Cost {cost} | {'Better' if better else 'Worse'} | gamma {self.true_gamma:.3}")
            if nan_flag or error_flag or lack_progress_flag:
                print("Reload")
                self.ns = int(self.ns_base + (np.random.rand() - 0.5) * 2 * self.nv)
                self.restore()
                continue
            else:
                self.loss_cmp = self.loss.item()
                self.visualize(i)
            if better:
                self.save()
                self.l2_min = l2
                best_epoch = i
            self.lossR.append([self.loss.item(),self.loss_pde.item(),self.loss_bc.item(),l2.item(),cost])
        self.lossR.save_as_csv(self.loss_target)
        self.lossR.plot(f'{self.target_dir}/l2_loss.png')
        print(f"Best Model at Epoch {best_epoch} with L2 {self.l2_min:.4e} | gamma is {self.weight_pde/self.weight_bc}")

import signal
import sys

def infer():
    od = cfg.output_dir
    gamma = 0.333
    wb = 1.0
    wp = 1.0
    Re = cfg.Re
    output_dir = f"{od}/Re{Re}gamma0"
    cfg.set_output_dir(output_dir)
    tsonn = LidCavity(output_dir,cfg.nb,cfg.ns,cfg.nv,Re,f"./ref/slant-{Re}g.csv",layers=cfg.layers,weight_bc=wb,weight_pde=wp,gamma = gamma)
    tsonn.restore()
    tsonn.visualize()

def train(gamma_bound):
    # od = cfg.output_dir
    od = "output"
    gamma = 0.33
    wb = 1.0
    wp = 1.0
    Re = cfg.Re
    output_dir = f"{od}/Re{Re}gamma_bound{gamma_bound}"
    cfg.set_output_dir(output_dir)
    tsonn = LidCavity(output_dir,cfg.nb,cfg.ns,cfg.nv,Re,f"./ref/slant-{Re}g.csv",layers=cfg.layers,weight_bc=wb,weight_pde=wp,gamma = gamma,gamma_bound=gamma_bound)
    def closure_c(signum, frame):
        tsonn.decontructor_hook()
        sys.exit(0)
    signal.signal(signal.SIGINT, closure_c)
    tsonn.train_adam(100,10)
    tsonn.train(epoch=cfg.num_iter)
    tsonn.visualize_best()

if __name__ == "__main__":
    for i in range(6,12):
        train(i*0.1)
