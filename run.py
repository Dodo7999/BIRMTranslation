import hydra
from omegaconf import DictConfig

from src.prepare.prepare_data import prepare_data
from src.train.train import execute


@hydra.main(config_path="config", config_name="config", version_base=None)
def train(cfg: DictConfig):
    #prepare_data(cfg)
    execute(cfg)

if __name__ == "__main__":
    train()
