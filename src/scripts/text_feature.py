from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import sys
import os
import torch
import pickle
import json
from Module.transformer import TransformerEncoder
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
model.to(device)

def func(prefix):
    with open(f'/data1/wjq/datasets/MELD/MELD.Raw/{prefix}_sent_emo_flatten.json', 'r') as f:
        file = json.load(f)
    res = []
    for text in file:
        encoded_input = tokenizer(text, return_tensors='pt').to(device)
        outputs = model(**encoded_input, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state
        features = hidden_states.mean(dim=1).squeeze().detach().cpu()
        res.append(features)
    print(len(res), res[0].shape)
    with open(f'/data1/wjq/datasets/MELD/MELD.Raw/{prefix}_sent_emo_flatten.pkl', 'wb') as f:
        pickle.dump(res, f)

if __name__=="__main__":
    func(prefix='train')
    func(prefix='test')
    func(prefix='dev')

# x = model(inputs_embeds=inputs_embeds, labels=labels)
# y = model(input_ids=input_ids, labels=labels_ids, output_hidden_states=True) # 有 label, 0 是 loss, 1 是 logits

sys.exit()


with open("/data1/wjq/datasets/MELD/MELD.Raw/test_audio.pkl", 'rb') as f:
    a = pickle.load(f)
    audio_feature = a[0].cpu()
    # print(audio_feature.shape) # [1, 113, 768]

with open("/data1/wjq/datasets/MELD/MELD.Raw/test_video.pkl", 'rb') as f:
    v = pickle.load(f)
    video_feature = v[0].cpu()
    # print(video_feature.shape) # [1, 197, 768]

fuse_feature = torch.cat((text_feature, video_feature, audio_feature), dim=1)
# print(fuse_feature.shape) # [1, 340, 768]

net = TransformerEncoder(embed_dim=768, num_heads=8, 
                         layers=3, attn_dropout=0.3, 
                         relu_dropout=0.1, res_dropout=0.1, 
                        embed_dropout=0.1, attn_mask=False)

T_A_V = net(fuse_feature)
print(T_A_V.shape)
out = net(T_A_V)
print(out.shape)
sys.exit()
T = T_A_V[:, :14, :]
print(T.shape)
print(T.equal(text_feature))
print(T)
print(text_feature)

# print(T_A_V.shape) # [1, 340, 768]
sys.exit()
output = model(inputs_embeds=fuse_feature, labels=torch.tensor(label_ids))
print(output[0])