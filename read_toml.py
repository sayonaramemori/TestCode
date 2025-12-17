import tomllib  # Python 3.11+
# import tomli as tomllib  # for Python <=3.10

class cfg():
    def __init__(self, table:str):
        settings = config[table]
        self.reynolds_num = settings['rey']
        self.num_pde = settings['num_pde']
        self.num_vibrate = settings['num_vibrate']
        self.num_bc = settings['num_bc']
        self.use_f64 = settings['use_f64']
        self.seed = settings['seed']
        self.outer_iteration = settings['outer_iteration']
        self.net_layers = settings['layers']
        self.output_dir = config['target']['dir']
        self.makedir_p(self.output_dir)
    @property
    def Re(self):
        return self.reynolds_num
    @property
    def nb(self):
        return self.num_bc
    @property
    def nv(self):
        return self.num_vibrate
    @property
    def ns(self):
        return self.num_pde
    @property
    def num_iter(self):
        return self.outer_iteration
    @property
    def layers(self):
        return self.net_layers
    def makedir_p(self,path):
        os.makedirs(path, exist_ok=True)
    def set_output_dir(self,path):
        os.makedirs(path, exist_ok=True)
        self.output_dir = path

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

import os
# target_dir = config['target']['dir']
# os.makedirs(target_dir, exist_ok=True)
os.makedirs('model', exist_ok=True)
os.makedirs('loss_record', exist_ok=True)

def get_cfg(table):
    return cfg(table)
