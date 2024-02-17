import numpy as np
from torch.utils.data import Dataset


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
