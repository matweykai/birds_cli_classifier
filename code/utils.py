import torch
import random
import numpy as np


RANDOM_SEED = 1234
loader_gen = torch.Generator()

def seed_worker(_):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

        
def config_randomness():
    global loader_gen
    
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(RANDOM_SEED)
    
    torch.cuda.manual_seed_all(RANDOM_SEED)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False