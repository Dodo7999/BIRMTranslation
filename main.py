import logging
import torch
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset
import numpy as np
import evaluate

logging.basicConfig(level=logging.INFO)

max_input_length = 56
max_target_length = 56
source_lang = "en"
target_lang = "ru"

device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
model_checkpoint = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    inputs = [ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class Loader(Dataset):
    def __init__(self, inputs, labels, tokenizer, max_length=56, transform=None, target_transform=None):
        self.labels = labels
        self.inputs = inputs

        self.tokenizer = tokenizer
        self.max_length = max_length

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


# raw_datasets_val = load_dataset('json', data_files={'train': ['eval.txt']})['train'].select(range(100))
raw_datasets_train = load_dataset("wmt19", "ru-en", split='train[:1000000]')
raw_datasets_val = load_dataset('json', data_files={'train': ['eval.txt']})['train']
datasets_train = raw_datasets_train.map(preprocess_function, batched=True)
datasets_val = raw_datasets_val.map(preprocess_function, batched=True)
train_inputs = np.array(datasets_train['input_ids'], dtype=object)
l = np.array([len(i) for i in train_inputs])
train_inputs = train_inputs[l < 150]
train_targets = np.array(datasets_train['labels'], dtype=object)
train_targets = train_targets[l < 150]
val_inputs = np.array(datasets_val['input_ids'], dtype=object)
l = np.array([len(i) for i in val_inputs])
val_inputs = val_inputs[l < 150]
val_targets = np.array(datasets_val['labels'], dtype=object)
val_targets = val_targets[l < 150]

n_epoch = 3
cel = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CyclicLR(opt, step_size_up=5000, mode='triangular2', cycle_momentum=False,
                                              base_lr=3e-6, max_lr=4e-4)

print(f"Count trainer data = {len(train_inputs)}")
print(f"Count eval data = {len(val_inputs)}")

butch_num = 20
google_bleu = evaluate.load("google_bleu", keep_in_memory=True)
train_loader = Loader(inputs=train_inputs, labels=train_targets, tokenizer=tokenizer)
eval_loader = Loader(inputs=val_inputs, labels=val_targets, tokenizer=tokenizer)
for i in range(n_epoch):
    model.train()
    gen = generator(train_loader, butch_num)
    index = 0
    for input_ids, attention_mask, decoder_input_ids, decoder_attention_mask in gen:
        logits = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                       decoder_attention_mask=decoder_attention_mask).logits
        loss = cel(logits.permute(0, 2, 1), decoder_input_ids.masked_fill(decoder_attention_mask != 1, -100))
        loss.backward()
        opt.step()
        opt.zero_grad()
        scheduler.step()

        if index % 50 == 0:
            t = torch.cuda.get_device_properties(device).total_memory / 1048576 / 1024
            r = torch.cuda.memory_reserved(device) / 1048576 / 1024
            a = torch.cuda.memory_allocated(device) / 1048576 / 1024
            f = r - a
            print(f"Count = {index * butch_num}, t = {t}, r = {r}, a = {a}, f = {f}")
            print(f"Epoch = {i}, loss = {loss}, batch_index = {index}")

        if index % 5000 == 0 and index > 0:
            with torch.no_grad():
                model.eval()
                gen = generator(eval_loader, butch_num)
                targets = []
                pred_seq = []
                for input_ids, attention_mask, decoder_input_ids, decoder_attention_mask in gen:
                    targets += tokenizer.batch_decode(decoder_input_ids)
                    pred_seq += tokenizer.batch_decode(
                        model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=150))
                print(google_bleu.compute(predictions=pred_seq, references=targets))

        index += 1
    print(f"Epoch = {i}")

    with torch.no_grad():
        model.eval()
        gen = generator(eval_loader, butch_num)
        targets = []
        pred_seq = []
        for input_ids, attention_mask, decoder_input_ids, decoder_attention_mask in gen:
            targets += tokenizer.batch_decode(decoder_input_ids)
            pred_seq += tokenizer.batch_decode(
                model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=150))
        print(pred_seq)
        print(google_bleu.compute(predictions=pred_seq, references=targets))
