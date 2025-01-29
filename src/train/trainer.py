import logging

import evaluate
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

log = logging.getLogger(__file__.split('/')[-1])


def generator(data, batch_size, shuffle=False):
    ids = np.arange(len(data))
    steps = len(data) // batch_size
    if len(data) % batch_size != 0:
        steps += 1
    if shuffle:
        np.random.shuffle(ids)
    for i in range(steps):
        batch_ids = ids[i * batch_size: (i + 1) * batch_size]
        yield data[batch_ids][0], data[batch_ids][1], data[batch_ids][2], data[batch_ids][3]


def generatorEnviroment(data_env, batch_size, batch_num, shuffle=False):
    ids = np.arange(batch_size * batch_num)
    if shuffle:
        np.random.shuffle(ids)
    for i in range(batch_num):
        batch_ids = ids[i * batch_size: (i + 1) * batch_size] % len(data_env[0])
        yield data_env[0][batch_ids], data_env[1][batch_ids], data_env[2][batch_ids], data_env[3][batch_ids]


def generatorWithEnviroment(data, batch_size, clusters, shuffle=False):
    unique_clusters = np.unique(clusters)
    size_biggest_cluster = 0
    for i in unique_clusters:
        size_biggest_cluster = max(len(clusters[clusters == i]), size_biggest_cluster)
    batch_num = size_biggest_cluster // batch_size
    env_gen = []
    for i in unique_clusters:
        env_gen.append(generatorEnviroment(data[clusters == i], batch_size, batch_num, shuffle))
    for i in range(batch_num):
        batch = []
        for j in range(len(unique_clusters)):
            batch.append(next(env_gen[j]))
        yield batch


class MyDataLoader:
    def __init__(self, loader, batch_size2, clusters=None, shuffle=False):
        self.loader = loader
        self.batch_size = batch_size2
        self.clusters = clusters
        self.shuffle = shuffle

    def __iter__(self):
        if self.clusters is None:
            return generator(self.loader, self.batch_size, self.shuffle)
        else:
            return generatorWithEnviroment(self.loader, self.batch_size, self.clusters, self.shuffle)


class Loader(Dataset):
    def __init__(self, inputs, tokenizer, test_dataset_size=None, is_val=False):
        self.inputs = inputs
        self.tokenizer = tokenizer
        self.is_val = is_val
        self.test_dataset_size = test_dataset_size

    def __len__(self):
        if not self.is_val:
            return len(self.inputs)
        return self.test_dataset_size

    def __getitem__(self, idx):
        inputs_list = self.inputs[idx]

        d = {
            "labels": "input_ids",
            "labels_attention_mask": "attention_mask"
        }
        input_features = {k: np.array(inputs_list[k])[:, 0] for k in ['input_ids', 'attention_mask']}
        label_features = {d[k]: np.array(inputs_list[k])[:, 0] for k in ['labels', 'labels_attention_mask']}

        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        with self.tokenizer.as_target_tokenizer():
            labels_batch = self.tokenizer.pad(
                label_features,
                padding=True,
                return_tensors="pt",
            )

        target_input_batch = labels_batch["input_ids"]
        if not self.is_val:
            target_input_batch[target_input_batch == self.tokenizer.pad_token_id] = -100
        target_attention_batch = labels_batch['attention_mask']

        input_batch = batch['input_ids']
        attention_bacth = batch['attention_mask']

        return input_batch, attention_bacth, target_input_batch, target_attention_batch


def evaluate_seq2seq_model(cfg, model, val_dataset, tokenizer):
    model.eval()

    bleu = evaluate.load("./bleu")
    bertscore = evaluate.load("./bertscore")
    bleu_results = []
    bertscore_results = {
        'precision': [],
        'recall': [],
        'f1': []
    }
    with torch.no_grad():
        val_loader = MyDataLoader(
            loader=Loader(
                inputs=val_dataset,
                tokenizer=tokenizer,
                test_dataset_size=cfg.train_params.test_dataset_size,
                is_val=True
            ),
            batch_size2=cfg.train_params.batch_size,
            shuffle=True
        )
        for input_batch, attention_batch, target_input_batch, target_attention_batch in tqdm(val_loader,
                                                                                             desc="Evaluating", position=0, leave=False):
            outputs = model(
                input_ids=input_batch.to(cfg.train_params.device),
                attention_mask=attention_batch.to(cfg.train_params.device),
                labels=target_input_batch.to(cfg.train_params.device)
            )
            logits = outputs.logits

            generated_tokens = logits.argmax(dim=-1)
            # log.info(generated_tokens.shape)
            # log.info(target_input_batch.shape)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(target_input_batch, skip_special_tokens=True)
            bertscore_result = bertscore.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                model_type="distilbert-base-uncased"
            )
            if len(bleu_results) < 3:
                log.info(decoded_preds)
                log.info(decoded_labels)
            bertscore_results['precision'] += bertscore_result['precision']
            bertscore_results['recall'] += bertscore_result['recall']
            bertscore_results['f1'] += bertscore_result['f1']
            bleu_results.append(
                bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])['bleu'])

    log.info(
        f"Bert Score: precition = {np.array(bertscore_results['precision']).mean()}, recall = {np.array(bertscore_results['recall']).mean()}, f1 = {np.array(bertscore_results['f1']).mean()}; BLEU: {np.array(bleu_results).mean()}")
    model.train()


def train_seq2seq_model(cfg, model, train_dataset, val_dataset, tokenizer):
    model.to(cfg.train_params.device)
    model.train()

    if cfg.shift.type != "None":
        train_clusters = np.fromfile(f"{cfg.shift.path}/clusters_train.dat", dtype=int)
        lambda_regularization = torch.tensor(1.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train_params.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=cfg.train_params.gamma)
    optimization_steps = 0
    for epoch in range(cfg.train_params.num_epochs):
        i = 0
        if cfg.shift.type != "None":
            train_loader = MyDataLoader(
                loader=Loader(inputs=train_dataset, tokenizer=tokenizer), clusters=train_clusters,
                batch_size2=cfg.train_params.batch_size, shuffle=True)

            total_loss = 0
            kommulative = 0
            progress = tqdm(
                train_loader,
                desc=f"Training Epoch {epoch + 1}/{cfg.train_params.num_epochs}, loss = {total_loss}, optimization steps = {optimization_steps}, lambda = {lambda_regularization}", position=0, leave=False
            )
            for envs in progress:
                i+=1
                kommulative += 1
                loss_list = []
                for input_batch, attention_batch, target_input_batch, target_attention_batch in envs:
                    result = model(
                        attention_mask=attention_batch.to(cfg.train_params.device),
                        input_ids=input_batch.to(cfg.train_params.device),
                        labels=target_input_batch.to(cfg.train_params.device),
                    )
                    labels = target_input_batch.to(cfg.train_params.device)
                    random_vector = torch.normal(mean=1, std=0.1, size=(result.logits.shape[2],)).to(cfg.train_params.device)
                    shift_logits = (result.logits.float() * random_vector.reshape(1, 1, -1))[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100, reduction="mean")
                    loss_list.append(loss)
                loss_t = torch.stack(loss_list)
                penalty = ((loss_t - loss_t.mean()) ** 2).sum()

                los = loss_t.sum()
                last_layers = list(model.children())[-1]
                los.backward(retain_graph=True)
                var_f = torch.std(last_layers.weight.grad.detach())
                optimizer.zero_grad()
                penalty.backward(retain_graph=True)
                var = torch.std(last_layers.weight.grad.detach())
                regularization = var_f / var
                optimizer.zero_grad()
                regularization = 0.2 * lambda_regularization + regularization
                if regularization < 10000.0:
                    lambda_regularization = regularization
                else:
                    lambda_regularization = torch.tensor(10000.0).to(cfg.train_params.device)

                loss = loss_t.sum() + lambda_regularization * penalty

                (loss / cfg.train_params.kommulation_steps).backward()
                total_loss += loss.detach().cpu().item() / cfg.train_params.kommulation_steps
                if kommulative == cfg.train_params.kommulation_steps:
                    optimization_steps += 1
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    kommulative = 0
                    progress.set_description(
                        f"Training Epoch {epoch + 1}/{cfg.train_params.num_epochs}, loss = {loss_t.clone().detach().cpu()}, optimization steps = {optimization_steps}, lambda = {regularization}")
                    total_loss = 0
                if i % cfg.train_params.eval_every_step == 0:
                    evaluate_seq2seq_model(cfg, model, val_dataset, tokenizer)
                    model.save_pretrained(cfg.model_save_path + f"/{cfg.model.type}_{optimization_steps}", from_pt=True)

        else:
            train_loader = MyDataLoader(
                loader=Loader(inputs=train_dataset, tokenizer=tokenizer),
                batch_size2=cfg.train_params.batch_size, shuffle=True)

            total_loss = 0
            kommulative = 0
            progress = tqdm(train_loader,
                            desc=f"Training Epoch {epoch + 1}/{cfg.train_params.num_epochs}, loss = {total_loss}, optimization steps = {optimization_steps}", position=0, leave=False)
            for input_batch, attention_batch, target_input_batch, target_attention_batch in progress:
                i += 1
                kommulative += 1
                outputs =  model(
                    attention_mask=attention_batch.to(cfg.train_params.device),
                    input_ids=input_batch.to(cfg.train_params.device),
                    labels=target_input_batch.to(cfg.train_params.device),
                )
                loss = outputs.loss

                (loss / cfg.train_params.kommulation_steps).backward()
                total_loss += loss.detach().cpu().item() / cfg.train_params.kommulation_steps
                if kommulative == cfg.train_params.kommulation_steps:
                    optimization_steps += 1
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    kommulative = 0
                    progress.set_description(
                        f"Training Epoch {epoch + 1}/{cfg.train_params.num_epochs}, loss = {total_loss}, optimization steps = {optimization_steps}")
                    total_loss = 0

                if i % cfg.train_params.eval_every_step == 0:
                    evaluate_seq2seq_model(cfg, model, val_dataset, tokenizer)
                    model.save_pretrained(cfg.model_save_path + f"/{cfg.model.type}_{optimization_steps}", from_pt=True)

    evaluate_seq2seq_model(cfg, model, val_dataset, tokenizer)
    model.save_pretrained(cfg.model_save_path + f"/{cfg.model.type}_final", from_pt=True)

    return model
