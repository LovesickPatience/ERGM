Follow the steps below to prepare the data, train the model, and run inference.

## 0. Installation

Before you begin, ensure you have the necessary dependencies installed.

### Prerequisites

- [FFmpeg](https://ffmpeg.org/): A powerful framework for handling multimedia files.
- [Montreal Forced Aligner (MFA)](https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html): Used to generate precise time alignments between audio and text.

### Python Dependencies

Clone this repository, navigate into the project directory, and install the required Python packages using `requirements.txt`.

```bash
git clone https://github.com/EmpatheticResponse/ERGM.git
cd ERGM
pip install -r requirements.txt
```

## 1. Data Preparation

The data preparation process consists of three main stages: downloading the datasets, pre-processing the files, and extracting features.

### Step 1: Download Datasets

Download the required datasets from their official sources:

- [MELD: Multimodal EmotionLines Dataset](https://affective-meld.github.io/)
- [IEMOCAP: Interactive Emotional Dyadic Motion Capture Database](https://sail.usc.edu/iemocap/)
- [MEDIC: A Multimodal Empathy Dataset in Counseling](https://ustc-ac.github.io/datasets/medic/)

### Step 2: Pre-processing

For each video, perform the following steps to create aligned data clips.

1.  **Extract Full Audio Track**: For eacg video, use FFmpeg to extract a 16kHz mono `.wav` audio file from each video.
    ```shell
    ffmpeg -i input_video.mp4 -vn -ar 16000 -ac 1 -c:a pcm_s16le output_audio.wav
    ```

2.  **Generate Timestamps**: Use the Montreal Forced Aligner (MFA) to perform a forced alignment. This will generate precise start and end timestamps for every word in the transcript.
    ```shell
    mfa align /path/to/your_corpus english_us_arpa english_us_arpa /path/to/output_folder
    ```

3.  **Segment Media Files**: Use the generated timestamps to segment the original video file into clips corresponding to each utterance.
    ```shell
    ffmpeg -i input_video.mp4 -ss time_start -to time_end -c copy output_clip.mp4
    ```
    After this step, you will have a tightly aligned `(text, video_clip, audio_clip)` triplet for each utterance.

4.  **Extract Keyframes**: Finally, extract keyframes from each video clip for later use.
    ```shell
    ffmpeg -i input_video.mp4 -vf "select='eq(pict_type,I)'" -vsync vfr output_folder/keyframe-%03d.jpg
    ```

### Step 3: Feature Extraction

Run the feature extraction script to process the pre-processed data.

```bash
python data_process/feature_extraction.py
```
> **Note**: You may need to adjust settings within the script (such as file paths) to match your environment.

### Step 4: Load Data

Execute the `load_data.sh` script to prepare the final data files for the model.

```shell
sh load_data.sh
```

After the script finishes, processed files will be generated in the `data/` directory with the following structure:
```
data
├--gpt2
│   ├── train_utters.pickle
│   ├── train_ids.pickle
│   ├── valid_utters.pickle
│   └── valid_ids.pickle
```

## 2. Training and Inference

### Training

To train the model, run the `train.sh` script. You can modify hyperparameters (e.g., learning rate, batch size) within the script as needed.

```bash
sh train.sh
```

### Inference

Use the `infer.sh` script to run inference with a trained model checkpoint. Be sure to specify the checkpoint name in the command.

```bash
sh infer.sh checkpoint_name
