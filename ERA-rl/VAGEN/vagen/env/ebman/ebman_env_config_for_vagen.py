from vagen.env.base.base_env_config import BaseEnvConfig

class EBManEnvConfig(BaseEnvConfig):
    def __init__(self, env_name):
        self.env_name = env_name
    
    def config_id(self) -> str:
        return self.env_name
    def get(self, key, default=None):
        return None
    def generate_seeds(self, env_size, train_size):
        import numpy as np
        import random
        import json

        np.random.seed(42)
        
        train_ranges = [(0,47), (200,247), (400,435)]
        train_candidates = []
        for start, end in train_ranges:
            train_candidates.extend(list(range(start, end)))
        train_seeds = np.random.choice(train_candidates, size=train_size, replace=True).tolist()

        
        test_seeds = list(range(100, 148)) + list(range(300, 348))
        return train_seeds + test_seeds