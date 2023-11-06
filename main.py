import logging
import torch
from transformers import MT5ForConditionalGeneration
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
import evaluate
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)

max_input_length = 128
max_target_length = 128
source_lang = "en"
target_lang = "ru"

device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
model_checkpoint = "google/mt5-base"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = MT5ForConditionalGeneration.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs


def generator(data, batch_size, shuffle=False):
    ids = np.arange(len(data))
    steps = len(data) // batch_size
    if len(data) % batch_size != 0:
        steps += 1
    if shuffle:
        np.random.shuffle(ids)
    for i in range(steps):
        batch_ids = ids[i * batch_size: (i + 1) * batch_size]
        yield data[batch_ids]


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
        target_attention_batch = labels_batch['attention_mask']

        input_batch = batch['input_ids']
        attention_bacth = batch['attention_mask']

        return input_batch, attention_bacth, target_input_batch, target_attention_batch


tokenizer_cluster = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model_cluster = BertModel.from_pretrained("bert-base-multilingual-cased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(encoded_input)
print(output.shape)

# raw_datasets_val = load_dataset('json', data_files={'train': ['eval.txt']})['train'].select(range(100))
raw_datasets_train = load_dataset("opus100", "en-ru", split='train[:1000000]')
raw_datasets_val = load_dataset('json', data_files={'train': ['eval.txt']})['train']
datasets_train = raw_datasets_train.map(preprocess_function, batched=True)
datasets_val = raw_datasets_val.map(preprocess_function, batched=True)
train_inputs = np.array(datasets_train['input_ids'], dtype=object)
train_targets = np.array(datasets_train['labels'], dtype=object)
val_inputs = np.array(datasets_val['input_ids'], dtype=object)
val_targets = np.array(datasets_val['labels'], dtype=object)

n_epoch = 3
cel = torch.nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.CyclicLR(opt, step_size_up=20000, mode='triangular2', cycle_momentum=False,
                                              base_lr=1e-6, max_lr=2e-4)

print(f"Count trainer data = {len(train_inputs)}")
print(f"Count eval data = {len(val_inputs)}")

batch_size = 20
cluster_loader = MyDataLoader(
    loader=Loader(inputs=train_inputs, labels=train_targets, tokenizer2=tokenizer_cluster),
    batch_size2=batch_size, shuffle=True)
clusters_prob = []
with torch.no_grad():
    model_cluster.eval()
    ind = 0
    for input_ids, attention_mask, decoder_input_ids, decoder_attention_mask in cluster_loader:
        probability = torch.nn.functional.softmax(model_cluster(input_ids).logits).detach().cpu().numpy()[:, 0].tolist()
        clusters_prob += probability
        if ind % 1000 == 0:
            print(len(clusters_prob))
        ind += 1
n_clusters = 4
clusters = KMeans(n_clusters=4).fit(np.array(clusters_prob).reshape(-1, 1)).labels_
batch_size = 5
google_bleu = evaluate.load("google_bleu", keep_in_memory=True)
train_loader = MyDataLoader(
    loader=Loader(inputs=train_inputs, labels=train_targets, tokenizer2=tokenizer),
    batch_size2=batch_size, clusters=clusters, shuffle=True)
eval_loader = MyDataLoader(
    loader=Loader(inputs=val_inputs, labels=val_targets, tokenizer2=tokenizer),
    batch_size2=batch_size, shuffle=True)
for i in range(n_epoch):
    model.train()
    index = 0
    # for input_ids, attention_mask, decoder_input_ids, decoder_attention_mask in train_loader:
    #     loss = model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         decoder_attention_mask=decoder_attention_mask,
    #         labels=decoder_input_ids,
    #     ).loss
    for envs in train_loader:
        loss_list = []
        penalty = 0
        for input_ids, attention_mask, decoder_input_ids, decoder_attention_mask in envs:
            loss_list.append(model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=decoder_input_ids,
            ).loss)
        loss_t = torch.stack(loss_list)
        penalty = ((loss_t - loss_t.mean()) ** 2).sum()
        loss = loss_t.sum() + penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 0.1)
        opt.step()
        opt.zero_grad()
        scheduler.step()

        if index * batch_size * n_clusters % 1000 == 0:
            print(f"Count = {index * batch_size * n_clusters}")
            print(f"Epoch = {i}, loss = {loss}, penalty = {penalty}, batch_index = {index}")

        if index * batch_size * n_clusters % 25000 == 0 and index > 0:
            with torch.no_grad():
                model.eval()
                targets = []
                pred_seq = []
                for input_ids2, attention_mask2, decoder_input_ids2, decoder_attention_mask2 in eval_loader:
                    targets += tokenizer.batch_decode(decoder_input_ids2, skip_special_tokens=True)
                    pred_seq += tokenizer.batch_decode(
                        model.generate(input_ids=input_ids2, attention_mask=attention_mask2,
                                       max_length=max_target_length), skip_special_tokens=True)
                for ind in range(10):
                    print(targets[ind])
                    print(pred_seq[ind])
                print(google_bleu.compute(predictions=pred_seq, references=targets))

        index += 1
    print(f"Epoch = {i}")

    with torch.no_grad():
        model.eval()
        targets = []
        pred_seq = []
        for input_ids2, attention_mask2, decoder_input_ids2, decoder_attention_mask2 in eval_loader:
            targets += tokenizer.batch_decode(decoder_input_ids2, skip_special_tokens=True)
            pred_seq += tokenizer.batch_decode(
                model.generate(input_ids=input_ids2, attention_mask=attention_mask2, max_length=max_target_length),
                skip_special_tokens=True)
        print(pred_seq)
        print(google_bleu.compute(predictions=pred_seq, references=targets))
