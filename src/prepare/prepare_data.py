from datasets import load_dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer


def prepare_data(cfg: DictConfig):
    ds = load_dataset(cfg.data_path, split='train[:10%]')
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path, local_files_only=True)

    def prepare_dataset_translate(examples, columns, tokenizer, length):
        input = tokenizer(
            examples[columns[0]],
            padding='max_length',
            truncation=True,
            max_length=length,
            return_tensors='pt'
        )

        label = tokenizer(
            examples[columns[1]],
            padding='max_length',
            truncation=True,
            max_length=length,
            return_tensors='pt',
        )
        input["labels"] = label.input_ids
        input["labels_attention_mask"] = label.attention_mask
        return input

    ds = ds.map(
        prepare_dataset_translate,
        batched=True,
        fn_kwargs={"columns": cfg.columns, "tokenizer": tokenizer, "length": 512}
    )

    ds = ds.train_test_split(test_size=0.01)
    ds["train"].save_to_disk(cfg.data_train)
    ds["test"].save_to_disk(cfg.data_test)
