from .base import BaseEnv
from .atari import AtariEnv
from omegaconf import OmegaConf
from src.common.class_utils import all_subclasses

ENVS = {subclass.get_name():subclass
        for subclass in all_subclasses(BaseEnv)}

def build_env(cfg):     
    cfg = OmegaConf.to_container(cfg)
    env_type = cfg.pop('type')
    env = ENVS[env_type]  
    train_env = env(**cfg)
    cfg['repeat_action_probability'] = 0.0
    eval_env = env(**cfg, episodic_lives=False)
    
    return train_env, eval_env
