import logging

import evaluate
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

log = logging.getLogger(__file__.split('/')[-1])


def generator(data, batch_size, shuffle=False):
    ids = np.arange(len(data))
    if shuffle:
        np.random.shuffle(ids)
    steps = (len(data) + batch_size - 1) // batch_size  # More efficient ceiling division
    for i in range(steps):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        batch_ids = ids[start_idx:end_idx]
        yield tuple(data[batch_ids][j] for j in range(4))  # More memory efficient


def generatorEnviroment(data_env, batch_size, batch_num, shuffle=False):
    data_len = len(data_env[0])
    ids = np.arange(batch_size * batch_num)
    if shuffle:
        np.random.shuffle(ids)
    for i in range(batch_num):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_ids = ids[start_idx:end_idx] % data_len
        yield tuple(data_env[j][batch_ids] for j in range(4))  # More memory efficient


def generatorWithEnviroment(data, batch_size, clusters, shuffle=False):
    unique_clusters = np.unique(clusters)
    size_biggest_cluster = max(len(clusters[clusters == i]) for i in unique_clusters)
    batch_num = (size_biggest_cluster + batch_size - 1) // batch_size  # More efficient ceiling division
    
    # Pre-compute cluster data to avoid repeated filtering
    cluster_data = {i: data[clusters == i] for i in unique_clusters}
    env_gen = [generatorEnviroment(cluster_data[i], batch_size, batch_num, shuffle) 
               for i in unique_clusters]
    
    for _ in range(batch_num):
        yield [next(gen) for gen in env_gen]


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
        # Pre-compute mapping dictionary
        self.feature_mapping = {
            "labels": "input_ids",
            "labels_attention_mask": "attention_mask"
        }
        # Pre-compute feature keys
        self.input_keys = ['input_ids', 'attention_mask']
        self.label_keys = ['labels', 'labels_attention_mask']

    def __len__(self):
        return self.test_dataset_size if self.is_val else len(self.inputs)

    def __getitem__(self, idx):
        inputs_list = self.inputs[idx]
        
        # More efficient dictionary comprehension
        input_features = {
            k: np.array(inputs_list[k])[:, 0] 
            for k in self.input_keys
        }
        label_features = {
            self.feature_mapping[k]: np.array(inputs_list[k])[:, 0] 
            for k in self.label_keys
        }

        # Process input features
        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            return_tensors="pt",
        )

        # Process label features
        with self.tokenizer.as_target_tokenizer():
            labels_batch = self.tokenizer.pad(
                label_features,
                padding=True,
                return_tensors="pt",
            )

        # Process target input batch
        target_input_batch = labels_batch["input_ids"]
        if not self.is_val:
            target_input_batch[target_input_batch == self.tokenizer.pad_token_id] = -100

        return (
            batch['input_ids'],
            batch['attention_mask'],
            target_input_batch,
            labels_batch['attention_mask']
        )


def evaluate_seq2seq_model(cfg, model, val_dataset, tokenizer):
    model.eval()
    
    # Load metrics once
    bleu = evaluate.load("./bleu")
    bertscore = evaluate.load("./bertscore")
    
    # Initialize accumulators for metrics
    metrics = {
        'bleu': [],
        'bertscore': {
            'precision': [],
            'recall': [],
            'f1': []
        }
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
        
        for input_batch, attention_batch, target_input_batch, target_attention_batch in tqdm(
            val_loader,
            desc="Evaluating",
            position=0,
            leave=False
        ):
            # Move tensors to device once
            input_batch = input_batch.to(cfg.train_params.device)
            attention_batch = attention_batch.to(cfg.train_params.device)
            target_input_batch = target_input_batch.to(cfg.train_params.device)
            
            # Get model outputs
            outputs = model(
                input_ids=input_batch,
                attention_mask=attention_batch,
                labels=target_input_batch
            )
            
            # Generate predictions
            generated_tokens = outputs.logits.argmax(dim=-1)
            
            # Decode predictions and labels
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(target_input_batch, skip_special_tokens=True)
            
            # Compute BERTScore
            bertscore_result = bertscore.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                model_type="distilbert-base-uncased"
            )
            
            # Log first 3 examples for debugging
            if len(metrics['bleu']) < 3:
                log.info(tokenizer.batch_decode(input_batch, skip_special_tokens=True))
                log.info(decoded_preds)
                log.info(decoded_labels)
            
            # Accumulate metrics
            metrics['bertscore']['precision'].extend(bertscore_result['precision'])
            metrics['bertscore']['recall'].extend(bertscore_result['recall'])
            metrics['bertscore']['f1'].extend(bertscore_result['f1'])
            metrics['bleu'].append(
                bleu.compute(
                    predictions=decoded_preds,
                    references=[[l] for l in decoded_labels]
                )['bleu']
            )
            
            # Clear CUDA cache if using GPU
            if cfg.train_params.device == 'cuda':
                torch.cuda.empty_cache()
    
    # Compute final metrics
    final_metrics = {
        'bertscore': {
            k: np.array(v).mean() for k, v in metrics['bertscore'].items()
        },
        'bleu': np.array(metrics['bleu']).mean()
    }
    
    log.info(
        f"Bert Score: precision = {final_metrics['bertscore']['precision']:.4f}, "
        f"recall = {final_metrics['bertscore']['recall']:.4f}, "
        f"f1 = {final_metrics['bertscore']['f1']:.4f}; "
        f"BLEU: {final_metrics['bleu']:.4f}"
    )
    
    model.train()
    return final_metrics


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
                    loss = model.loss_function(logits=shift_logits, labels=shift_labels, vocab_size=model.config.vocab_size)
                    # loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100, reduction="mean")
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
                    optimizer.zero_grad()
                    scheduler.step()
                    kommulative = 0
                    progress.set_description(
                        f"Training Epoch {epoch + 1}/{cfg.train_params.num_epochs}, loss = {total_loss}, optimization steps = {optimization_steps}")
                    total_loss = 0
                # print(next(model.model.layers[0].parameters()))
                if i % cfg.train_params.eval_every_step == 0:
                    evaluate_seq2seq_model(cfg, model, val_dataset, tokenizer)
                    model.save_pretrained(cfg.model_save_path + f"/{cfg.model.type}_{optimization_steps}", from_pt=True)

    evaluate_seq2seq_model(cfg, model, val_dataset, tokenizer)
    model.save_pretrained(cfg.model_save_path + f"/{cfg.model.type}_final", from_pt=True)

    return model
