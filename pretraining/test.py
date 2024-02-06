from corus import load_taiga_arzamas

from pretraining.preprocessing import mask_spans

# def split_string(input_str, chunk_size):
#     return [input_str[i:i + chunk_size] for i in range(0, len(input_str), chunk_size)]
#
#
# chunk_size = 1000
#
# path = 'taiga/Arzamas.tar.gz'
# records = load_taiga_arzamas(path)
# print(len(split_string(next(records).text, chunk_size)))
# print(len(split_string(next(records).text, chunk_size)))
# print(len(split_string(next(records).text, chunk_size)))
# print(len(split_string(next(records).text, chunk_size)))
# print(len(split_string(next(records).text, chunk_size)))
# print(len(split_string(next(records).text, chunk_size)))
# print(len(split_string(next(records).text, chunk_size)))
# print(len(split_string(next(records).text, chunk_size)))
# print(len(split_string(next(records).text, chunk_size)))
# print(len(split_string(next(records).text, chunk_size)))
# print(len(split_string(next(records).text, chunk_size)))
# print(len(split_string(next(records).text, chunk_size)))
# print(len(split_string(next(records).text, chunk_size)))
# print(len(split_string(next(records).text, chunk_size)))
# print(len(split_string(next(records).text, chunk_size)))

from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel

model_checkpoint = "ai-forever/rugpt3small_based_on_gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(tokenizer)
model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
print(model.config)
model = GPT2LMHeadModel(config=model.config)


model_inputs = tokenizer('I enjoy walking with my cute dog', return_tensors='pt')

# generate 40 new tokens
greedy_output = model.generate(**model_inputs, max_new_tokens=40)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))