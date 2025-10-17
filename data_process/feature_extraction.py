from typing import List

import torch
import librosa
from PIL import Image
from transformers import (Wav2Vec2Processor, Wav2Vec2Model, BlipProcessor, BlipModel, BartForConditionalGeneration,
                          BartTokenizer, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel,
                          BartTokenizerFast, BartModel, RobertaTokenizerFast)


# Text
class TextEncoder:
    def __init__(self, task: str, egc_model: str = 'facebook/bart-base', cls_model: str = 'roberta-base',
                 device: str = 'cpu'):
        self.task = task
        self.device = device
        if task == 'EGC':
            if BartTokenizerFast is None or BartModel is None:
                raise ImportError("transformers (Bart) not available.")
            self.tok = BartTokenizerFast.from_pretrained(egc_model)
            self.enc = BartModel.from_pretrained(egc_model).to(device)
            self.enc.eval()
        else:
            if RobertaTokenizerFast is None or RobertaModel is None:
                raise ImportError("transformers (RoBERTa) not available.")
            self.tok = RobertaTokenizerFast.from_pretrained(cls_model)
            self.enc = RobertaModel.from_pretrained(cls_model).to(device)
            self.enc.eval()

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        # Returns (B, D)
        enc = self.tok(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
        outputs = self.enc(**enc)
        if self.task == 'EGC':
            # Mean-pool encoder last_hidden_state (B, L, D)
            last = outputs.last_hidden_state
            mask = enc['attention_mask'].unsqueeze(-1)  # (B, L, 1)
            summed = (last * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            mean_pooled = summed / counts
            return mean_pooled.cpu()
        else:
            # Use [CLS] token (first token) for RoBERTa
            return outputs.last_hidden_state[:, 0].cpu()

# Audio
def extract_audio_features(audio_path):

    try:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    except Exception as e:
        return None

    try:
        audio_input, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        return None

    inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt")

    with torch.no_grad():
        features = model(**inputs).last_hidden_state

    return features


# Visual
def extract_image_features(image_path):

    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
    except Exception as e:
        return None
    

    try:
        raw_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return None

    inputs = processor(images=raw_image, return_tensors="pt")

    with torch.no_grad():

        vision_outputs = model.vision_model(**inputs)
        image_features = vision_outputs.last_hidden_state
    # shape: (batch_size, sequence_length, hidden_size)

    return image_features



if __name__ == "__main__":
    import numpy as np
    import soundfile as sf
    
    # replace with your audio path
    audio_features = extract_audio_features("audio.wav")
    if audio_features is not None:
        mean_audio_features = torch.mean(audio_features, dim=1)
        # save your audio features

    # replace with your image path
    image_features = extract_image_features("image.jpg")
    if image_features is not None:
        mean_image_features = torch.mean(image_features, dim=1)
        # save your image features
