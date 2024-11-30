from transformers import DataCollatorForLanguageModeling, AutoTokenizer, \
    MT5ForConditionalGeneration, Qwen2ForCausalLM, Qwen2Tokenizer

import torch

def get_model_pretrained(path, type):
    if type == "mt5":
        pretrained = MT5ForConditionalGeneration.from_pretrained(path, local_files_only=True)
        new_model = MT5ForConditionalGeneration(pretrained.config)
        new_model.shared = pretrained.shared
        return new_model
    else:
        pretrained = Qwen2ForCausalLM.from_pretrained(path, local_files_only=True)
        config = pretrained.config
        config.hidden_size = 896
        config.intermediate_size = 1024
        new_model = Qwen2ForCausalLM(config)
        new_model.model.embed_tokens = pretrained.model.embed_tokens
        return new_model


def get_tokenizer_pretrained(path):
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def get_collator_pretrained(tokenizer, type):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=16)
