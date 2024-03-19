import gc
import logging
import re

from torch.nn import CrossEntropyLoss
from transformers import  GPT2LMHeadModel
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import numpy as np
import evaluate
from sklearn.cluster import KMeans
from corus import load_taiga_arzamas, load_taiga_fontanka, load_taiga_interfax, load_taiga_kp, load_taiga_nplus1, \
    load_taiga_lenta, load_taiga_social, load_taiga_stihi, load_taiga_proza
import torch
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


def compute_perplexity(
        predictions, model, tokenizer, batch_size: int = 16, add_start_token: bool = True, device=None,
        max_length=None
):
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        assert (
                len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        assert (
                tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        predictions,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    ).to(device)

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in range(0, len(encoded_texts), batch_size):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index]
        attn_mask = attn_masks[start_index:end_index]

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

        shift_logits = out_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp(
            (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls += perplexity_batch.tolist()

    return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}


bertscore = load("bertscore")
perplexity = load("perplexity", module_type="metric")
logging.basicConfig(level=logging.INFO)

max_input_length = 128
max_target_length = 128
chunk_size = 1000

device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
# torch.set_default_device(device)
model_checkpoint = "ai-forever/rugpt3small_based_on_gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(tokenizer)
print(tokenizer.eos_token)
print(tokenizer.bos_token)
tokenizer.padding_side = "right"
model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
config = model.config
config.n_embd = 504
print(config)
model = GPT2LMHeadModel(config=config).to(device)
torch.save(model, "/userspace/bma/BIRMTranslation/model_base.pth")

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

train_set = []
val_set = []
clusters = []
for i, path in enumerate(paths):
    records = path[1](path[0])
    for record in records:
        if record.text != '' and len(train_set) < 200_000 * (i + 1):
            text = record.text

            texts_p = text.split("\n")
            texts_s = re.split("\.|\n", text)

            for text_s in texts_s:
                if text_s != "" and len(text_s):
                    train_set.append(tokenizer.bos_token + text_s + tokenizer.eos_token)
                    clusters.append(len(text_s))

            for text_p in texts_p:
                if text_p != "" and len(text_p):
                    train_set.append(tokenizer.bos_token + text_p + tokenizer.eos_token)
                    clusters.append(len(text_p))
clusters = np.array(clusters)
clusters = KMeans(n_clusters=3).fit(clusters.reshape(-1, 1)).labels_
train_set = list(map(preprocess_function, train_set))
train_set = np.array(train_set, dtype=object)
some_set = []
some_clusters = []
for i, text_tokens in enumerate(train_set):
    if len(text_tokens[0]) <= 1000:
        some_set.append(text_tokens)
        some_clusters.append(clusters[i])
train_set = np.array(some_set, dtype=object)
clusters = np.array(some_clusters)
cls_size = []
for i in range(3):
    cls_size.append(len(train_set[clusters == i][0][0]))
cls_size = np.array(cls_size)
print(cls_size)
min_cls = cls_size.argmin()
max_cls = cls_size.argmax()
mid_cls = np.argsort(cls_size)[len(cls_size) // 2]
print(f"Clusters shape = {clusters.shape}")
current_cluster_test = max_cls
envs_train = []
envs_eval = []
envs_test = []
clusters_train = []
clusters_eval = []
clusters_test = []
cl_unic = np.unique(clusters)
for cluster in cl_unic:
    samples = train_set[clusters == cluster]
    cluster_samples = clusters[clusters == cluster]
    samples_len = len(samples)
    if cluster == current_cluster_test:
        envs_test.append(samples[:2000])
        clusters_test.append(cluster_samples[:2000])
    else:
        envs_train.append(samples[:int(samples_len * 0.8)])
        envs_eval.append(samples[int(samples_len * 0.8):int(samples_len * 0.9)][:2000])
        clusters_eval.append(cluster_samples[int(samples_len * 0.8):int(samples_len * 0.9)][:2000])
        envs_test.append(samples[int(samples_len * 0.9):][:2000])
        clusters_test.append(cluster_samples[int(samples_len * 0.9):][:2000])
train_set = np.concatenate(envs_train, axis=0)
val_set = np.concatenate(envs_eval, axis=0)
val_clusters = np.concatenate(clusters_eval, axis=0)
test_set = np.concatenate(envs_test, axis=0)
test_clusters = np.concatenate(clusters_test, axis=0)
train_inputs = train_set[:, 0]
train_targets = train_set[:, 1]
val_inputs = val_set[:, 0]
val_targets = val_set[:, 1]
test_inputs = test_set[:, 0]
test_targets = test_set[:, 1]
gc.collect()
n_epoch = 5
cel = torch.nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.999996)

print(f"Count trainer data = {len(train_inputs)}")
print(f"Count trainer data = {len(val_inputs)}")
print(f"Count test data = {len(test_inputs)}")

batch_size = 4
google_bleu = evaluate.load("google_bleu", keep_in_memory=True)
for i in range(n_epoch):
    model.train()
    index = 0
    train_loader = MyDataLoader(
        loader=Loader(inputs=train_inputs, labels=train_targets, tokenizer2=tokenizer),
        batch_size2=batch_size*2, shuffle=True)
    val_loader = MyDataLoader(
        loader=Loader(inputs=val_inputs, labels=val_targets, tokenizer2=tokenizer),
        batch_size2=batch_size*2, clusters=val_clusters, shuffle=False)
    jk = 0
    for input_ids, attention_mask in train_loader:
        model.train()
        loss = model(
            attention_mask=attention_mask.to(device),
            input_ids=input_ids.to(device),
            labels=input_ids.to(device),
        ).loss
        loss.backward()
        if jk == 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            opt.zero_grad()
            scheduler.step()
            jk = 0
        jk += 1

        if index % 1000 == 0 and index > 0:
            print(f"Count = {index}")
            print(f"Epoch = {i}, loss = {loss}, batch_index = {index}, lr = {opt.param_groups[0]['lr']}")
            if index % 100_000 == 0:
                model.eval()
                torch.save(model, f"/userspace/bma/BIRMTranslation/model_base_{index}.pth")
                perplexity = [0, 0]
                count = [0, 0]
                length = [0, 0]
                for env in val_loader:
                    j = 0
                    for input_ids2, attention_mask2 in env:
                        count[j] += 1
                        pred_seq = tokenizer.batch_decode(
                            model.generate(
                                input_ids=input_ids2[:, :2].to(device),
                                max_new_tokens=min(input_ids2.shape[1] + 10, 2048),
                                min_new_tokens=input_ids2.shape[1],
                                repetition_penalty=5.0
                            )
                        )
                        perplexity[j] += compute_perplexity(
                            predictions=pred_seq,
                            model=model,
                            tokenizer=tokenizer,
                            device=device
                        )['mean_perplexity']
                        if j == 0 or j == 1:
                            length[j] = input_ids2.shape[1]
                            if count[j] == 1:
                                print(pred_seq)
                        j += 1
                for j in range(2):
                    print(f"Perplexity env {j} = {perplexity[j] / max(count[j], 1)}, length = {length[j]}")

        index += 1
    print(f"Epoch = {i}")

    if index == 301000:
        test_loader = MyDataLoader(
            loader=Loader(inputs=test_inputs, labels=test_targets, tokenizer2=tokenizer),
            batch_size2=batch_size, clusters=test_clusters, shuffle=True)
        model.eval()
        perplexity = [0, 0, 0]
        count = [0, 0, 0]
        length = [0, 0, 0]
        for env in test_loader:
            j = 0
            for input_ids2, attention_mask2 in env:
                count[j] += 1
                pred_seq = tokenizer.batch_decode(
                    model.generate(
                        input_ids=input_ids2[:, :2].to(device),
                        max_new_tokens=min(input_ids2.shape[1] + 10, 2048),
                        min_new_tokens=input_ids2.shape[1],
                        repetition_penalty=5.0
                    )
                )
                perplexity[j] += compute_perplexity(
                    predictions=pred_seq,
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )['mean_perplexity']
                if j == 0 or j == 1 or j == 2:
                    length[j] = input_ids2.shape[1]
                    if count[j] == 1:
                        print(pred_seq)
                j += 1
        for j in range(3):
            print(f"Perplexity env {j} = {perplexity[j] / max(count[j], 1)}, length = {length[j]}")


    test_loader = MyDataLoader(
        loader=Loader(inputs=test_inputs, labels=test_targets, tokenizer2=tokenizer),
        batch_size2=batch_size, clusters=test_clusters, shuffle=True)
    model.eval()
    perplexity = [0, 0, 0]
    count = [0, 0, 0]
    length = [0, 0, 0]
    for env in test_loader:
        j = 0
        for input_ids2, attention_mask2 in env:
            count[j] += 1
            pred_seq = tokenizer.batch_decode(
                model.generate(
                    input_ids=input_ids2[:, :2].to(device),
                    max_new_tokens=min(input_ids2.shape[1] + 10, 2048),
                    min_new_tokens=input_ids2.shape[1],
                    repetition_penalty=5.0
                )
            )
            perplexity[j] += compute_perplexity(
                predictions=pred_seq,
                model=model,
                tokenizer=tokenizer,
                device=device
            )['mean_perplexity']
            if j == 0 or j == 1 or j == 2:
                length[j] = input_ids2.shape[1]
                if count[j] == 1:
                    print(pred_seq)
            j += 1
    for j in range(3):
        print(f"Perplexity env {j} = {perplexity[j] / max(count[j], 1)}, length = {length[j]}")

torch.save(model, f"/userspace/bma/BIRMTranslation/model_base_final.pth")
