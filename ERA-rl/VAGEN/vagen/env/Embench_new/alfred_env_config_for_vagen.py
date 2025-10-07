from vagen.env.base.base_env_config import BaseEnvConfig

class AlfredEnvConfig(BaseEnvConfig):
    def __init__(self, env_name):
        self.env_name = env_name
    
    def config_id(self) -> str:
        return self.env_name
    def get(self, key, default=None):
        return None