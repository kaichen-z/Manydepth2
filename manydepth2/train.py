import os 
import torch
import random
import numpy as np
from .trainer import Trainer
from .trainer_hr_flow_0 import Trainer as Trainer2
from .options import MonodepthOptions
import logging

def seed_all(seed):
    if not seed:
        seed = 1
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

options = MonodepthOptions()
opts = options.parse()
seed_all(opts.pytorch_random_seed)

logging.basicConfig(level=logging.INFO, 
                    format='%(message)s',
                    handlers=[logging.FileHandler(f'logs/{opts.model_name}.txt'), 
                              logging.StreamHandler()])

if __name__ == "__main__":
    if opts.mode == 'many':
        trainer = Trainer(opts)
        trainer.train()
    if opts.mode == 'many2':
        trainer = Trainer2(opts)
        trainer.train()
