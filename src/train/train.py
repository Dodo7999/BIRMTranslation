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

    model = get_model_pretrained(cfg.model_path, cfg.model_type)
    tokenizer = get_tokenizer_pretrained(cfg.tokenizer_path)
    model = freeze_emb(model, cfg.model_type)

    train_seq2seq_model(
        model=model,
        train_dataset=train,
        val_dataset=test,
        tokenizer=tokenizer,
    )
    #
    # trainer.save_model(cfg.new_model_path)
    # if not os.path.exists("reports"):
    #     os.makedirs("reports")
    # plt.gcf().set_size_inches(10, 5)
    # plt.plot([hist["epoch"] for hist in trainer.state.log_history if "loss" in hist],
    #          [hist["loss"] for hist in trainer.state.log_history if "loss" in hist],
    #          label='Train loss')
    # plt.plot([hist["epoch"] for hist in trainer.state.log_history if "eval_loss" in hist],
    #          [hist["eval_loss"] for hist in trainer.state.log_history if "eval_loss" in hist],
    #          label='Test loss')
    # plt.legend()
    # plt.savefig('reports/learning_curve_loss.png', bbox_inches='tight')
    #
    # plt.plot([hist["epoch"] for hist in trainer.state.log_history if "eval_recall" in hist],
    #          [hist["eval_recall"] for hist in trainer.state.log_history if "eval_recall" in hist],
    #          label='Test recall')
    # plt.legend()
    # plt.savefig('reports/learning_curve_recall_valid.png', bbox_inches='tight')
    #
    # plt.plot([hist["epoch"] for hist in trainer.state.log_history if "eval_precision" in hist],
    #          [hist["eval_precision"] for hist in trainer.state.log_history if "eval_precision" in hist],
    #          label='Test precision')
    # plt.legend()
    # plt.savefig('reports/learning_curve_precision_valid.png', bbox_inches='tight')
    #
    # plt.plot([hist["epoch"] for hist in trainer.state.log_history if "eval_f1" in hist],
    #          [hist["eval_f1"] for hist in trainer.state.log_history if "eval_f1" in hist],
    #          label='Test f1')
    # plt.legend()
    # plt.savefig('reports/learning_curve_f1_valid.png', bbox_inches='tight')
    #
    # plt.plot([hist["epoch"] for hist in trainer.state.log_history if "eval_f1" in hist],
    #          [hist["eval_f1"] for hist in trainer.state.log_history if "eval_f1" in hist],
    #          label='Test loss')
    # plt.legend()
    # plt.savefig('reports/learning_curve_f1_valid.png', bbox_inches='tight')
    #
    # metrics = trainer.evaluate()
