import json
import os
from tqdm import tqdm
from transformers import GPT2Tokenizer

DATA_DIRECTORY = "path/to/your/data/directory" 
DATA_PREFIXES = ['train', 'valid', 'test'] 


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

sp1_token = "<sp1>"
sp2_token = "<sp2>"
eos_token = tokenizer.eos_token
bos_token = "<bos>"
audio_token = "<aud>"
image_token = "<img>"
cap_bos_token = "<cap_bos>"
cap_eos_token = "<cap_eos>"
sadness, joy, disgust, fear, anger, neutral, surprise = "<sadness>", "<joy>", \
        "<disgust>", "<fear>", "<anger>", "<neutral>", "<surprise>"

special_tokens = {
    'additional_special_tokens': [bos_token, sp1_token, sp2_token, 
                                  image_token, audio_token,
                                  cap_bos_token, cap_eos_token,
                                  sadness, joy, disgust, fear, anger, neutral, surprise,
                                  ]}

num_new_tokens = tokenizer.add_special_tokens(special_tokens)



for prefix in DATA_PREFIXES:
    
    input_file_path = os.path.join(DATA_DIRECTORY, f'{prefix}_sent_emo.json')
    output_file_path = os.path.join(DATA_DIRECTORY, f'{prefix}_sent_emo_ids.json')
    
    if not os.path.exists(input_file_path):
        print(f"Warning: Input file '{input_file_path}' not found, skipping.")
        continue

    print(f"Processing file: {input_file_path}...")
    
    ids = []
    with open(input_file_path, 'r', encoding='utf-8') as json_file:
        dialogues = json.load(json_file)

    for dialogue in tqdm(dialogues, desc=f"Tokenizing {prefix} data"):
        dialogue_ids = []
        for utter in dialogue:
            # utter[0] contains the text to be tokenized
            tokens = tokenizer.tokenize(utter[0])
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            dialogue_ids.append(token_ids)
        ids.append(dialogue_ids)
    
    # Verify that the number of processed items matches the original number of dialogues
    assert len(ids) == len(dialogues)

    # Write the processed token IDs to a new JSON file
    print(f"Saving processed IDs to: {output_file_path}...")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(ids, f)