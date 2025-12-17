import time
import numpy as np
import torch
from tool_box import visual_3d
from tool_box import *
from read_toml import *

cfg = get_cfg('settings-3d')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
use_f64 = cfg.use_f64
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
    def __init__(self,output_dir,nb,ns,nv,Re,ref='./points_4000.csv',layers=[2, 64, 128, 256, 128, 64, 3],dtau = None,weight_pde = 1.0,weight_bc = 4.0,l2_min = 1.0,gamma_bound=1.0):
        self.gamma = 0.2
        self.gamma_bound = gamma_bound
        self.target_dir = output_dir
        self.weight_bc = weight_bc
        self.weight_pde = weight_pde
        self.loss = self.loss_pde = self.loss_bc = torch.tensor([0.0])
        self.Re = Re
        self.model_path = f'{output_dir}/slant3d-re{Re}.pth'
        self.loss_target = f'{output_dir}/slant3d-re{Re}.csv'
        self.nb = nb
        self.ns = ns
        self.nv = nv
        self.iter = 500
        self.ns_base = ns
        self.l2_min = l2_min
        self.loss_cmp = 0.99
        self.dtau = 0.5 if dtau == None else dtau
        self.device = DEVICE
        self.model = MLP(layers).to(self.device)
        self.sample_interior(ns)
        self.sample_boundary(nb)
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=500,
            history_size=100,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=0
        )
        points = np.loadtxt(ref,delimiter=',') #read the sliced_data

        self.wall_xy_p = points[points[:,0]==0.8]
        self.wall_xyz = self.wall_xy_p[:,0:3]
        self.wall_p = self.wall_xy_p[:,-1]

        self.xyz_ref = points[:,0:3]
        self.p_ref = points[:,-1]
        self.uv_ref = points[:,3:5]
        ref_V_mag = np.linalg.norm(self.uv_ref,axis=1).reshape(-1,1)
        self.X_ref = torch.from_numpy(self.xyz_ref.astype(ntype)).to(DEVICE)
        self.V_ref = torch.from_numpy(ref_V_mag.astype(ntype)).to(DEVICE)
        self.lossR = LossRecorder(col=['loss','pde_loss','bc_loss','L2','cost'])
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
                    # Y[i][j] = 1.0
                    Y[i][j] = 0.15 - 0.5*X[i][j]
        self.XY_meshgrid = (X,Y)
        self.XY_meshgrid_origin = np.meshgrid(x, y)
        
    def save(self):
        torch.save(self.model.state_dict(),self.model_path)

    def restore(self,path=None):
        # self.model = torch.load(path,weights_only=False)
        path = self.model_path if path == None else path
        self.model.load_state_dict(torch.load(path))
        self.save()

    def sample_interior(self,n:int):
        xyz_slant = transform_2d_3d(sample_interior(int(n*0.05),dim=3),np.array([ [0.3,0,0], [0,0.15,0], [0,0,0.2] ]))
        idx = []
        for i in range(xyz_slant.__len__()):
            y_temp = -0.5*xyz_slant[i][0] + 0.15
            if y_temp - xyz_slant[i][1] >= 1e-6:
                continue
            idx.append(i)
        xyz_slant = xyz_slant[idx]
        xyz_left = transform_2d_3d(sample_interior(int(n*0.32),dim=3),np.array([ [0.3,0,0], [0,0.85,0], [0,0,0.2] ]),np.array([0,0.15,0]))
        xyz_right = transform_2d_3d(sample_interior(int(n*0.63),dim=3),np.array([ [0.5,0,0], [0,1,0], [0,0,0.2] ]),np.array([0.3,0,0]))
        xyz = np.concatenate([xyz_slant,xyz_left,xyz_right])
        self.xy_f = torch.tensor(xyz, dtype=ttype, device=DEVICE)

    def sample_boundary(self,n:int):
        top = transform_2d_3d(sample_interior(int(n*0.07)),np.array([ [0.8,0,0], [0,0,0.2] ]),np.array([0,1,0]))
        bot = transform_2d_3d(sample_interior(int(n*0.05)),np.array([ [0.5,0,0], [0,0,0.2] ]),np.array([0.3,0,0]))
        right = transform_2d_3d(sample_interior(int(n*0.08)),np.array([ [0,1,0], [0,0,0.2] ]),np.array([0.8,0,0]))
        left = transform_2d_3d(sample_interior(int(n*0.09)),np.array([ [0,0.85,0], [0,0,0.2] ]),np.array([0,0.15,0]))
        slant = transform_2d_3d(sample_interior(int(n*0.03)),np.array([ [0.3,-0.15,0], [0,0,0.2] ]),np.array([0,0.15,0]))
        front= transform_2d_3d(sample_slant(int(0.35*n)),np.array([ [1,0,0], [0,1,0] ]),np.array([0,0,0.2]))
        back= transform_2d_3d(sample_slant(int(0.35*n)),np.array([ [1,0,0], [0,1,0] ]))
        # xyz = np.concatenate([top,bot,right,left,slant,front,back]); show_samples_3d(xyz)

        self.top = torch.tensor(top,dtype=ttype,device=DEVICE)
        self.bottom = torch.tensor(bot,dtype=ttype,device=DEVICE)
        self.left = torch.tensor(left,dtype=ttype,device=DEVICE)
        self.right = torch.tensor(right,dtype=ttype,device=DEVICE)
        self.slant= torch.tensor(slant,dtype=ttype,device=DEVICE)
        self.front = torch.tensor(front,dtype=ttype,device=DEVICE)
        self.back = torch.tensor(back,dtype=ttype,device=DEVICE)

        begin = 0; end = begin + top.__len__();self.idx_top = slice(begin,end)
        begin = end; end = begin + bot.__len__();self.idx_bot = slice(begin,end)
        begin = end; end = begin + left.__len__();self.idx_left = slice(begin,end)
        begin = end; end = begin + right.__len__();self.idx_right = slice(begin,end)
        begin = end; end = begin + slant.__len__();self.idx_slant = slice(begin,end)
        begin = end; end = begin + front.__len__();self.idx_front = slice(begin,end)
        begin = end; end = begin + back.__len__();self.idx_back= slice(begin,end)

        self.boundary = torch.cat([self.top,self.bottom,self.left,self.right,self.slant,self.front,self.back])

    def TimeStepping(self):
        X = self.xy_f
        pred = self.model(X)
        u = pred[:,0:1];  v = pred[:,1:2];   w = pred[:,2:3]; p = pred[:,3:4]; 
        self.U0 = torch.cat([u,v,w,p]).detach()

    def pde_residuals(self):
        X = self.xy_f
        X.requires_grad_(True)
        pred = self.model(X)

        u = pred[:,0:1];  v = pred[:,1:2];   w = pred[:,2:3]; p = pred[:,3:4]; 
        u_xyz = fwd_gradients(u, X)
        v_xyz = fwd_gradients(v, X)
        w_xyz = fwd_gradients(w, X)
        p_xyz = fwd_gradients(p, X)
        u_x = u_xyz[:,0:1]; u_y = u_xyz[:,1:2]; u_z = u_xyz[:,2:3]
        v_x = v_xyz[:,0:1]; v_y = v_xyz[:,1:2]; v_z = v_xyz[:,2:3]
        w_x = w_xyz[:,0:1]; w_y = w_xyz[:,1:2]; w_z = w_xyz[:,2:3]
        p_x = p_xyz[:,0:1]; p_y = p_xyz[:,1:2]; p_z = p_xyz[:,2:3]
        u_xx = fwd_gradients(u_x, X)[:,0:1]; u_yy = fwd_gradients(u_y, X)[:,1:2]; u_zz = fwd_gradients(u_z, X)[:,2:3]
        v_xx = fwd_gradients(v_x, X)[:,0:1]; v_yy = fwd_gradients(v_y, X)[:,1:2]; v_zz = fwd_gradients(v_z, X)[:,2:3]
        w_xx = fwd_gradients(w_x, X)[:,0:1]; w_yy = fwd_gradients(w_y, X)[:,1:2]; w_zz = fwd_gradients(w_z, X)[:,2:3]

        res_u = u*u_x + v*u_y + w*u_z + p_x - 1.0/self.Re*(u_xx + u_yy + u_zz)
        res_v = u*v_x + v*v_y + w*v_z + p_y - 1.0/self.Re*(v_xx + v_yy + v_zz)
        res_w = u*w_x + v*w_y + w*w_z + p_z - 1.0/self.Re*(w_xx + w_yy + w_zz)
        res_rho = u_x + v_y + w_z
        U1 = torch.cat([u,v,w,p])
        R1 = torch.cat([res_u,res_v,res_w,res_rho])
        
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

    def p_anchor_residual(self):
        p_anchor = torch.tensor([[0.3,0.0],[0.0,0.15]],dtype=ttype,device=DEVICE)
        p_anchor.requires_grad_(True)
        pred = self.model(p_anchor)
        p = pred[:,2:3];
        return p + torch.tensor([[0.022385,0.56031]],dtype=ttype,device=DEVICE)

    def bc_residuals_soft(self):
        with torch.set_grad_enabled(True):
            self.boundary.requires_grad_(True)
            out = self.model(self.boundary)
            u = out[:, 0:1]; v = out[:, 1:2]; w = out[:, 2:3]
            bc_top_u = u[self.idx_top] - 0.0; bc_top_v = v[self.idx_top] - 0.0; bc_top_w = w[self.idx_top] - 0.0
            bc_bot_u = u[self.idx_bot] - 0.0; bc_bot_v = v[self.idx_bot] - 0.0; bc_bot_w = v[self.idx_bot] - 0.0
            bc_left_u = u[self.idx_left] - 0.0; bc_left_v = v[self.idx_left] - 0.0; bc_left_w = w[self.idx_left] - 0.0
            bc_right_u = u[self.idx_right] - 0.0; bc_right_v = v[self.idx_right] - 0.0; bc_right_w = w[self.idx_right] - 0.0
            # bc_slant_u = u[self.idx_slant] - 2.67; bc_slant_v = v[self.idx_slant] + 1.335 ; bc_slant_w = w[self.idx_slant] + 0.0
            # bc_slant_u = u[self.idx_slant] - 1.78; bc_slant_v = v[self.idx_slant] + 0.89 ; bc_slant_w = w[self.idx_slant] + 0.0
            bc_slant_u = u[self.idx_slant] - 0.89; bc_slant_v = v[self.idx_slant] + 0.445 ; bc_slant_w = w[self.idx_slant] + 0.0
            bc_front_u = u[self.idx_front] - 0.0; bc_front_v = v[self.idx_front] - 0.0; bc_front_w = w[self.idx_front] - 0.0
            bc_back_u = u[self.idx_back] - 0.0; bc_back_v = v[self.idx_back] - 0.0; bc_back_w = w[self.idx_back] - 0.0
            bc_u = torch.cat([bc_top_u, bc_bot_u, bc_left_u, bc_right_u, bc_slant_u, bc_front_u, bc_back_u], dim=0)
            bc_v = torch.cat([bc_top_v, bc_bot_v, bc_left_v, bc_right_v, bc_slant_v, bc_front_v, bc_back_v], dim=0)
            bc_w = torch.cat([bc_top_w, bc_bot_w, bc_left_w, bc_right_w, bc_slant_w, bc_front_w, bc_back_w], dim=0)
            return bc_u, bc_v, bc_w

    def train_adam(self,epoch=100,iter_n=3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.995)
        for _ in range(iter_n):
            self.sample_interior(self.ns)
            self.sample_boundary(self.nb)
            self.TimeStepping()
            for ep in range(1, epoch):
                self.model.train()
                optimizer.zero_grad()
                loss_pde = self.pde_residuals()
                bc_u, bc_v ,bc_w= self.bc_residuals_soft()
                loss_bc = (bc_u.pow(2).mean() + bc_v.pow(2).mean()+ bc_w.pow(2).mean())

                loss = self.weight_pde * loss_pde + self.weight_bc * loss_bc
                loss.backward()
                optimizer.step()
                scheduler.step()
                l2 = self.cal_L2()
                if ep%100 == 0:
                    print(f"[Adam] Epoch {ep:5d} | loss={loss.item():.4e} | pde={loss_pde.item():.4e} | bc={loss_bc.item():.4e} | l2={l2:.4e}")
            self.visualize(index="Adam-pretrain")

    def decontructor_hook(self):
        self.lossR.save_as_csv(f"./loss_record/loss-slant-re{self.Re}.csv")
        self.lossR.plot(f'{self.target_dir}/l2_loss.png')
        self.visualize_l2()

    def visualize_l2(self,path = None):
        self.model.load_state_dict(torch.load(self.model_path if path == None else path))
        self.model.eval()
        self.visualize(index='best')
        visual_ref(self.xyz_ref,self.uv_ref,f"{self.target_dir}/ref_uv.png")

    def visualize(self,index=None):
        target = f"{self.target_dir}/{index}"
        os.makedirs(target, exist_ok=True)
        visual_3d.speed_mag(self.XY_meshgrid,self.model,f"{target}/u.png",resol=self.resol,use_f64=use_f64)
        visual_3d.stream_line(self.XY_meshgrid_origin,self.model,f"{target}/s.png",resol=self.resol,use_f64=use_f64)
        visual_3d.pressure(self.XY_meshgrid,self.model,f"{target}/p.png",resol=self.resol,use_f64=use_f64)
        visual_3d.speed_l2_error(self.model,self.xyz_ref,self.uv_ref,f"{target}/u_l2.png",use_f64=use_f64)
        visual_3d.pressure_l2_error(self.model,self.xyz_ref,self.p_ref,f"{target}/p_l2.png",use_f64=use_f64)
        visual_pwall(self.model,self.wall_xyz,self.wall_p,f"{self.target_dir}/pp.png",use_double=use_f64)

    def train(self,epoch=300):
        LBFGS_step = 1
        best_epoch = 0
        for i in range(epoch):
            if self.gamma < self.gamma_bound:
                self.gamma += 0.001
            t0 = time.time()
            def closure_for_lbfgs():
                nonlocal LBFGS_step
                self.model.train()
                self.optimizer.zero_grad()
                self.loss_pde = self.pde_residuals()
                bc_u, bc_v ,bc_w= self.bc_residuals_soft()
                self.loss_bc = (bc_u.pow(2).mean() + bc_v.pow(2).mean() + bc_w.pow(2).mean())
                gm = self.gamma * (self.loss_pde.item()/self.loss_bc.item())
                self.weight_pde = 1.0 * gm
                self.weight_bc = 1.0 - self.weight_pde
                self.loss = self.weight_pde * self.loss_pde +  self.weight_bc * self.loss_bc
                self.loss.backward()
                LBFGS_step += 1
                return self.loss
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), max_iter=self.iter)
            self.TimeStepping()
            self.optimizer.step(closure_for_lbfgs)

            cost = int(time.time()-t0)
            l2 = self.cal_L2(); better = self.l2_min > l2
            nan_flag = not torch.isfinite(self.loss); lack_progress_flag = cost <= 9; error_flag = self.loss.item() > 4.5 * self.loss_cmp
            print(f"Epoch {i:3d}| ns {self.ns} |Re {self.Re}| L2 {l2:.4e} | PDE {self.loss_pde.item():.4e} | BC {self.loss_bc.item():.4e}| wp {self.weight_pde:.2} | Cost {cost} | {'Better' if better else 'Worse'}")
            if nan_flag or error_flag or lack_progress_flag:
                print("Reload")
                self.ns = int(self.ns_base + (np.random.rand() - 0.5) * 2 * self.nv)
                # self.dtau = 0.5 + (np.random.rand() - 0.5) * 2 * 0.25
                self.sample_boundary(self.nb)
                self.sample_interior(self.ns)
                self.restore()
                continue
            else:
                self.loss_cmp = self.loss.item()
            if better or i<15:
                self.save()
                self.l2_min = l2
                self.visualize(i)
                best_epoch = i
            self.lossR.append([self.loss.item(),self.loss_pde.item(),self.loss_bc.item(),l2.item(),cost])
        self.lossR.save_as_csv(self.loss_target)
        self.lossR.plot(f'{self.target_dir}/l2_loss.png')
        print(f"Best Model at Epoch {best_epoch} with L2 {self.l2_min:.4e}, Gb is {self.gamma_bound}")

import signal
import sys
def train(gamma_bound):
    Re = cfg.Re
    cfg.set_output_dir(f'output3d/Re{Re}gammb{gamma_bound}')
    tsonn = LidCavity(cfg.output_dir,cfg.nb,cfg.ns,cfg.nv,Re,f"./ref/slant-{Re}-3d-uniform.csv",layers=cfg.layers,weight_pde=1,weight_bc=1,dtau=.25,gamma_bound=gamma_bound)
    def closure_c(signum, frame):
        tsonn.decontructor_hook()
        sys.exit(0)
    signal.signal(signal.SIGINT, closure_c)
    tsonn.train_adam(200,4)
    tsonn.train(epoch=cfg.num_iter)
    tsonn.visualize_l2()

if __name__ == "__main__":
    for i in range(5,7):
        train(i*0.1)
