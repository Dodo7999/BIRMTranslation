import numpy as np
from datasets import load_dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import os


def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()


def prepare_data(cfg: DictConfig):
    ds = load_dataset(cfg.data.path, split='train[:1%]')
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.path, local_files_only=True)
    if cfg.shift.type == "clusters":
        tokenizer_b = AutoTokenizer.from_pretrained('models/rubert', local_files_only=True)
        model_b = AutoModel.from_pretrained('models/rubert', local_files_only=True)
        model_b.to(cfg.train_params.device)

    def prepare_dataset_translate(examples, columns, tokenizer, length, shift):
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

        if shift == "clusters":
            embeddings = embed_bert_cls(examples[columns[0]], model_b, tokenizer_b)
            input["embed"] = embeddings
        else:
            input["len"] = len(examples[columns[0]])
        return input

    ds = ds.map(
        prepare_dataset_translate,
        batched=False,
        fn_kwargs={
            "columns": [cfg.data.source_col, cfg.data.target_col],
            "tokenizer": tokenizer,
            "length": 512,
            "shift": cfg.shift.type
        }
    )

    if not os.path.exists(cfg.shift.path):
        os.makedirs(cfg.shift.path)


    inds = list(range(len(ds)))


    if cfg.shift.type == "length":
        clusters = KMeans(n_clusters=3).fit(np.array(ds["len"]).reshape(-1, 1)).labels_
        inds_train, inds_test = train_test_split(inds, test_size=0.1, random_state=42)
        clusters[inds_train].tofile(f"{cfg.shift.path}/clusters_train.dat")
        clusters[inds_test].tofile(f"{cfg.shift.path}/clusters_test.dat")
    elif cfg.shift.type == "clusters":
        clusters = KMeans(n_clusters=3).fit(np.array(ds["embed"])).labels_
        inds_train, inds_test = train_test_split(inds, test_size=0.1, random_state=42)
        clusters[inds_train].tofile(f"{cfg.shift.path}/clusters_train.dat")
        clusters[inds_test].tofile(f"{cfg.shift.path}/clusters_test.dat")
    else:
        inds_train, inds_test = train_test_split(inds, test_size=0.1, random_state=42)

    ds.select(inds_train).save_to_disk(cfg.data_train)
    ds.select(inds_test).save_to_disk(cfg.data_test)
