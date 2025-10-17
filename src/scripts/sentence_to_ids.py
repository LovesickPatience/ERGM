import json
from tqdm import tqdm
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

sp1_token = "<sp1>"
sp2_token = "<sp2>"
eos_token = tokenizer.eos_token
bos_token = "<bos>"
cap_bos_token = "<bos_cap>"
cap_eos_token = "<eos_cap>"
audio_token = "<aud>"
image_token = "<img>"

special_tokens = {
    'additional_special_tokens': [bos_token, sp1_token, sp2_token, 
                                    cap_bos_token, cap_eos_token, 
                                    image_token, audio_token]}

num_new_tokens = tokenizer.add_special_tokens(special_tokens)


ids = []
file_name = '../../data/microsoft/DialoGPT-medium/multi_test_sent_emo.json'
json_file = open(file_name)
dialogues = json.load(json_file)


for dialogue in tqdm(dialogues):
    dialogue_ids = []
    for utter in dialogue:
        tokens = tokenizer.tokenize(utter)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        dialogue_ids.append(token_ids)
    ids.append(dialogue_ids)

assert len(ids) == len(dialogues)

with open(f"../../data/microsoft/DialoGPT-medium/multi_test_sent_emo_ids.json", 'w') as f:
    json.dump(ids, f)