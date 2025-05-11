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

    # if hasattr(cfg, 'train_size') and cfg.train_size > 0:
    #     train = train.select(range(min(cfg.train_size, len(train))))
    #     log.info(f"Limited training dataset to {len(train)} examples")
    
    # if hasattr(cfg, 'val_size') and cfg.val_size > 0:
    #     test = test.select(range(min(cfg.val_size, len(test))))
    #     log.info(f"Limited validation dataset to {len(test)} examples")

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
