import logging
import re

import torch
from transformers import MT5ForConditionalGeneration, GPT2Model, GPT2LMHeadModel
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import numpy as np
import evaluate
from sklearn.cluster import KMeans
from corus import load_taiga_arzamas, load_taiga_fontanka, load_taiga_interfax, load_taiga_kp, load_taiga_nplus1, \
    load_taiga_lenta, load_taiga_social, load_taiga_stihi, load_taiga_proza
import torch
from transformers import BertTokenizer, BertForMaskedLM
from evaluate import load


def generator(data, batch_size, shuffle=False):
    ids = np.arange(len(data))
    steps = len(data) // batch_size
    if len(data) % batch_size != 0:
        steps += 1
    if shuffle:
        np.random.shuffle(ids)
    for i in range(steps):
        batch_ids = ids[i * batch_size: (i + 1) * batch_size]
        yield data[batch_ids][0], data[batch_ids][1]


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
    def __init__(self, inputs, labels, tokenizer2):
        self.labels = labels
        self.inputs = inputs

        self.tokenizer = tokenizer2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs_list = self.inputs[idx]
        labels_list = self.labels[idx]

        input_features = [{"input_ids": feature} for feature in inputs_list]
        label_features = [{"input_ids": feature} for feature in labels_list]
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
        target_input_batch[target_input_batch == self.tokenizer.pad_token_id] = -100
        target_attention_batch = labels_batch['attention_mask']

        input_batch = batch['input_ids']
        attention_bacth = batch['attention_mask']

        return input_batch, attention_bacth, target_input_batch, target_attention_batch


bertscore = load("bertscore")
perplexity = load("perplexity", module_type="metric")
logging.basicConfig(level=logging.INFO)

max_input_length = 128
max_target_length = 128
chunk_size = 1000

device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
model_checkpoint = "ai-forever/rugpt3small_based_on_gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(tokenizer)
print(tokenizer.eos_token)
print(tokenizer.bos_token)
tokenizer.padding_side = "right"
model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
print(model.config)
model = GPT2LMHeadModel(config=model.config)


def wer(hypothesis, reference):
    hypothesis = "".join(hypothesis)
    reference = "".join(reference)

    hypothesis = hypothesis.lower()
    reference = reference.lower()

    errors = 0
    for h, r in zip(hypothesis, reference):
        if h != r:
            errors += 1

    # Вычисление WER
    wer = errors / len(reference)

    return wer


def preprocess_function(examples):
    inputs = examples
    model_inputs = tokenizer(inputs, text_target=inputs)
    return [model_inputs['input_ids'], model_inputs['labels']]


paths = [
    ['/userspace/bma/BIRMTranslation/taiga/Arzamas.tar.gz', load_taiga_arzamas],
    ['/userspace/bma/BIRMTranslation/taiga/Interfax.tar.gz', load_taiga_interfax],
    ['/userspace/bma/BIRMTranslation/taiga/KP.tar.gz', load_taiga_kp],
    ['/userspace/bma/BIRMTranslation/taiga/Lenta.tar.gz', load_taiga_lenta],
    ['/userspace/bma/BIRMTranslation/taiga/NPlus1.tar.gz', load_taiga_nplus1],
    ['/userspace/bma/BIRMTranslation/taiga/proza_ru.zip', load_taiga_proza],
    ['/userspace/bma/BIRMTranslation/taiga/Fontanka.tar.gz', load_taiga_fontanka],
    ['/userspace/bma/BIRMTranslation/taiga/social.tar.gz', load_taiga_social],
    ['/userspace/bma/BIRMTranslation/taiga/stihi_ru.zip', load_taiga_stihi],
]

# Загрузка и кластеризацию по кластерам
# train_set = []
# val_set = []
# clusters = []
# clusters_val = []
# for i, path in enumerate(paths):
#     records = path[1](path[0])
#     cluster = []
#     cluster_val = []
#     for record in records:
#         if record.text != '' and len(cluster) < 200_000:
#             text = record.text[:1000]
#             if text != '' and len(val_set) > (i + 1) * 5:
#                 cluster.append(i)
#                 train_set.append([text, text])
#             elif text != '':
#                 cluster_val.append(i)
#                 val_set.append([text[:50], text[:100]])
#     clusters.append(cluster)
#     clusters_val.append(cluster_val)
# print(len(train_set))
# clusts = []
# train_set = train_set
# for clust in clusters:
#     clusts += clust
# clusters = np.array(clusts)
# print(clusters.shape)
# train_set = list(map(preprocess_function, train_set))
# train_inputs = np.array(train_set, dtype=object)[:, 0]
# train_targets = np.array(train_set, dtype=object)[:, 1]
#
# print(len(val_set))
# val_set = val_set
# clusts_val = []
# for clust in clusters_val:
#     clusts_val += clust
# clusters_val = np.array(clusts_val)
# print(clusters_val.shape)
# val_set = list(map(preprocess_function, val_set))
# val_inputs = np.array(val_set, dtype=object)[:, 0]
# val_targets = np.array(val_set, dtype=object)[:, 1]

# Загрузка и кластеризацию по длинне
train_set = []
val_set = []
clusters = []
for i, path in enumerate(paths):
    records = path[1](path[0])
    for record in records:
        if record.text != '' and len(train_set) < 750_000 * (i+1):
            text = record.text

            texts_p = text.split("\n")
            texts_s = re.split("\.|\n", text)

            for text_s in texts_s:
                if text_s != "":
                    train_set.append(tokenizer.bos_token + text_s + tokenizer.eos_token)
                    clusters.append(len(text_s))
                    # if len(text_s) < 100:
                    #     clusters.append(0)
                    # elif len(text_s) < 500:
                    #     clusters.append(1)
                    # else:
                    #     clusters.append(2)

            for text_p in texts_p:
                if text_p != "":
                    train_set.append(tokenizer.bos_token + text_p + tokenizer.eos_token)
                    clusters.append(len(text_p))
                    # if len(text_p) < 100:
                    #     clusters.append(0)
                    # elif len(text_p) < 500:
                    #     clusters.append(1)
                    # else:
                    #     clusters.append(2)

print(len(train_set))
clusters = np.array(clusters)
print(clusters.shape)
clusters = KMeans(n_clusters=3).fit(clusters.reshape(-1, 1)).labels_
train_set = list(map(preprocess_function, train_set))
train_set = np.array(train_set, dtype=object)
len_cls = []
for cls in np.unique(clusters):
    len_cls.append(len(clusters[clusters == cls]))
inds = 2
for i in range(3):
    if len_cls[i] != max(len_cls) and len_cls != min(len_cls):
        inds = i
len_val = len(train_set[clusters == inds])
val_set = train_set[clusters == inds][len_val // 2:len_val // 2 + 40]
train_set = np.hstack((train_set[clusters != inds], train_set[clusters == inds][:len_val // 10]))
clusters = np.hstack((clusters[clusters != inds], clusters[clusters == inds][:len_val // 10]))
train_inputs = train_set[:, 0]
train_targets = train_set[:, 1]
val_inputs = val_set[:, 0]
val_targets = val_set[:, 1]

n_epoch = 60
cel = torch.nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.99999)

print(f"Count trainer data = {len(train_inputs)}")
print(f"Count trainer data = {len(val_inputs)}")

batch_size = 10
google_bleu = evaluate.load("google_bleu", keep_in_memory=True)
for i in range(n_epoch):
    model.train()
    index = 0
    train_loader = MyDataLoader(
        loader=Loader(inputs=train_inputs, labels=train_targets, tokenizer2=tokenizer),
        batch_size2=batch_size, clusters=clusters, shuffle=True)
    val_loader = MyDataLoader(
        loader=Loader(inputs=val_inputs, labels=val_targets, tokenizer2=tokenizer),
        batch_size2=2, shuffle=False)
    for envs in train_loader:
        model.train()
        loss_list = []
        for input_ids, attention_mask in envs:
            loss_list.append(model(
                attention_mask=attention_mask,
                input_ids=input_ids,
                labels=input_ids,
            ).loss)
        loss_t = torch.stack(loss_list)
        penalty = ((loss_t - loss_t.mean()) ** 2).sum()
        loss = loss_t.sum() + penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        opt.step()
        opt.zero_grad()
        scheduler.step()

        if index % 1000 == 0:
            print(f"Count = {index}")
            print(
                f"Epoch = {i}, loss = {loss}, losses = {loss_t.detach().tolist()}, penalty = {penalty}, batch_index = {index}, lr = {opt.param_groups[0]['lr']}")
            if index % 10000 == 0:
                model.eval()
                for input_ids2, attention_mask2 in val_loader:
                    dec = input_ids2
                    dec[dec == -100] = tokenizer.pad_token_id
                    targets = tokenizer.batch_decode(dec)
                    pred_seq = tokenizer.batch_decode(
                        model.generate(input_ids=input_ids2[:, :2]))
                    print(targets)
                    print(pred_seq)
                    print(google_bleu.compute(predictions=pred_seq, references=targets))
                    print(wer(pred_seq, targets))
                    print(bertscore.compute(predictions=pred_seq, references=targets,
                                            model_type="bert-base-multilingual-cased"))
                    print(perplexity.compute(predictions=pred_seq, model_id='gpt2'))

        index += 1
    print(f"Epoch = {i}")
