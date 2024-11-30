import evaluate
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np


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
        yield data_env[0][batch_ids], data_env[1][batch_ids]


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
    def __init__(self, inputs, tokenizer, is_val = False):
        self.inputs = inputs
        self.tokenizer = tokenizer
        self.is_val = is_val

    def __len__(self):
        if not self.is_val:
            return len(self.inputs)
        return 100

    def __getitem__(self, idx):
        inputs_list = self.inputs[idx]
        d = {
            "labels": "input_ids",
            "labels_attention_mask": "attention_mask"
        }
        input_features = {k: inputs_list[k] for k in ['input_ids','attention_mask']}
        label_features = {d[k]: inputs_list[k] for k in ['labels','labels_attention_mask']}

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


def evaluate_seq2seq_model(model, val_dataset, tokenizer, batch_size, device):
    model.eval()

    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    bleu_results = []
    bertscore_results = {
        'precision': [],
        'recall': [],
        'f1': []
    }
    with torch.no_grad():
        val_loader = MyDataLoader(
            loader=Loader(inputs=val_dataset, tokenizer=tokenizer, is_val=True),
            batch_size2=batch_size, shuffle=True)
        for input_batch, attention_batch, target_input_batch, target_attention_batch in tqdm(val_loader, desc="Evaluating"):

            outputs = model(
                input_ids=input_batch.to(device),
                attention_mask=attention_batch.to(device),
                labels=target_input_batch.to(device)
            )
            logits = outputs.logits

            generated_tokens = logits.argmax(dim=-1)

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(target_input_batch, skip_special_tokens=True)
            bertscore_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels,
                                        model_type="distilbert-base-uncased")
            print(decoded_preds)
            print(decoded_labels)
            bertscore_results['precision'] += bertscore_result['precision']
            bertscore_results['recall'] += bertscore_result['recall']
            bertscore_results['f1'] += bertscore_result['f1']
            bleu_results.append(bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])['bleu'])

    print(f"Bert Score: precition = {np.array(bertscore_results['precision']).mean()}, recall = {np.array(bertscore_results['recall']).mean()}, f1 = {np.array(bertscore_results['f1']).mean()}; BLEU: {np.array(bleu_results).mean()}")
    model.train()


def train_seq2seq_model(model, train_dataset, val_dataset, tokenizer, num_epochs=3, batch_size=2, learning_rate=5e-4,
                        device='cuda'):
    # Переключить модель в режим обучения
    model.to(device)
    model.train()

    # Оптимизатор
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.999996)

    # Цикл обучения
    for epoch in range(num_epochs):
        i = 0
        train_loader = MyDataLoader(
            loader=Loader(inputs=train_dataset, tokenizer=tokenizer),
            batch_size2=batch_size, shuffle=True)
        total_loss = 0
        kommulative = 0
        progress = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}, loss = {total_loss}")
        for input_batch, attention_batch, target_input_batch, target_attention_batch in progress:
            i+=1
            kommulative+=1


            # Прямой проход (forward pass)
            outputs = model(
                input_ids=input_batch.to(device),
                attention_mask=attention_batch.to(device),
                labels=target_input_batch.to(device)
            )
            loss = outputs.loss

            # Обратный проход (backward pass) и шаг оптимизации
            (loss / 16.0).backward()
            total_loss += loss.detach().cpu().item() / 16
            if kommulative == 16:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                scheduler.step()
                # Обнуление градиентов
                optimizer.zero_grad()
                kommulative = 0
                progress.set_description(f"Training Epoch {epoch + 1}/{num_epochs}, loss = {total_loss}")
                total_loss = 0

            if i % 20000 == 0:
                evaluate_seq2seq_model(model, val_dataset, tokenizer, batch_size, device)
                model.save_pretrained("C:\\Users\\brat_\\PycharmProjects\\BIRMTranslation\\model_checkpoints", from_pt=True)

        evaluate_seq2seq_model(model, val_dataset, tokenizer, batch_size, device)
        model.save_pretrained("C:\\Users\\brat_\\PycharmProjects\\BIRMTranslation\\model_checkpoints", from_pt=True)

    return model
