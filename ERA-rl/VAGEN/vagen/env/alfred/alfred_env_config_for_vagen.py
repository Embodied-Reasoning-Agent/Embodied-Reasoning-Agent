from vagen.env.base.base_env_config import BaseEnvConfig
import random
import json

class AlfredEnvConfig(BaseEnvConfig):
    def __init__(self, image_mode, image_interval):
        self.image_mode = image_mode
        self.image_interval = image_interval
    
    def config_id(self) -> str:
        return f"{self.image_mode}_{self.image_interval}"
    def get(self, key, default=None):
        return None
    def generate_seeds(self, env_size, train_size):
        import numpy as np
        import json

        # vanilla data generation
        block = list(range(0, 50)) + list(range(300, 350)) + list(range(400, 450))
        train_seeds = []
        for _ in range(100):
            group = block.copy()
            random.shuffle(group)
            train_seeds.extend(group)
        print("Generated train seeds:", train_seeds[:10], "...")  # Print first 10 seeds for verification

        test_seeds = list(range(100, 150)) + list(range(200, 250))

        
        return train_seeds + test_seeds