#!~miniconda3/envs/rl/bin python
import hydra
from omegaconf import DictConfig, OmegaConf
import Environments
import Src.Algorithms.Agent as Agent
from Environments.Gridworld_687 import Gridworld_687
from time import time
import torch
import numpy as np

class Solver:
    def __init__(self, config):
        
        self.config = config
        self.env = self.Get_Environment(config)
        self.state_dim = np.shape(self.env.reset())[0]

    def Get_Environment(self, config):
        env_name = config.env.name

        if env_name == "Gridworld_687":
            return Gridworld_687() #TODO: pass config params
        

@hydra.main(config_name="config")
def main(config : DictConfig) -> None:
    t = time()

    solver = Solver(config)

    print("Total time taken: {}".format(time()-t))


if __name__ == "__main__":
    main()