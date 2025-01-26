from transformers import DataCollatorForLanguageModeling, AutoTokenizer, \
    MT5ForConditionalGeneration, Qwen2ForCausalLM, Qwen2Tokenizer

import torch
import torch.nn as nn

@torch.jit.script
def fused_sin(x):
    return torch.sin(x)



class Sine(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return fused_sin(input)

def get_model_pretrained(path, type):
    if type == "mt5":
        pretrained = MT5ForConditionalGeneration.from_pretrained(path, local_files_only=True)
        config = pretrained.config
        config.num_layers = 4
        config.num_decoder_layers = 4
        print(config)
        new_model = MT5ForConditionalGeneration(config)
        new_model.shared = pretrained.shared
        print(new_model)
        return new_model
    else:
        pretrained = Qwen2ForCausalLM.from_pretrained(path, local_files_only=True)
        config = pretrained.config
        config.hidden_size = 896
        config.intermediate_size = 1024
        config.num_hidden_layers = 12
        new_model = Qwen2ForCausalLM(config)
        for layer in new_model.model.layers:
            layer.mlp.act_fn = Sine()
        new_model.model.embed_tokens = pretrained.model.embed_tokens
        return new_model


def get_tokenizer_pretrained(path):
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def get_collator_pretrained(tokenizer, type):
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=16)
