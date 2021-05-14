#!~miniconda3/envs/rl/bin python
import hydra
from omegaconf import DictConfig, OmegaConf
import Src.Algorithms.Agent as Agent
from time import time

class Solver:
    def __init__(self, config):
        
        self.config = config
        self.env = self.config.env
        

@hydra.main(config_name="config")
def main(config : DictConfig) -> None:
    t = time()

    solver = Solver(config)

    print("Total time taken: {}".format(time()-t))


if __name__ == "__main__":
    main()