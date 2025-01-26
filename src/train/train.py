import logging
from datasets import load_from_disk
from omegaconf import DictConfig

from src.train.trainer import train_seq2seq_model
from src.utils.freeze_embedings import freeze_emb
from src.utils.initialize import get_model_pretrained, get_tokenizer_pretrained

log = logging.getLogger("Train")


def execute(cfg: DictConfig):
    train = load_from_disk(cfg.data_train)
    test = load_from_disk(cfg.data_test)

    model = get_model_pretrained(cfg.model.path, cfg.model.type)
    tokenizer = get_tokenizer_pretrained(cfg.model.path)
    model = freeze_emb(model, cfg.model.type)

    train_seq2seq_model(
        cfg = cfg,
        model=model,
        train_dataset=train,
        val_dataset=test,
        tokenizer=tokenizer,
    )
