#!~miniconda3/envs/rl/bin python
import hydra
from omegaconf import DictConfig, OmegaConf
import Src.Algorithms.Agent as Agent

@hydra.main(config_name="config")
def main(cfg : DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    

if __name__ == "__main__":
    main()