import os
import sys
import pickle as pkl
import argparse
import numpy as np
import pandas as pd

# Define global mappings and lists that are part of the data specification, not personal info.
EMOTION_LIST = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMOTION_TO_ID = {emo: i for i, emo in enumerate(EMOTION_LIST)}
SENTIMENT_LIST = ["neutral", "positive", "negative"]
SENTIMENT_TO_ID = {senti: i for i, senti in enumerate(SENTIMENT_LIST)}

def process_dialogues_from_csv(csv_path):
    """
    Reads a CSV file, groups utterances by dialogue ID, and extracts emotion and sentiment labels.

    Args:
        csv_path (str): The full path to the input CSV file.

    Returns:
        dict: A dictionary containing lists of emotion and sentiment labels, grouped by dialogue.
              e.g., {"emotion": [[...], [...]], "sentiment": [[...], [...]]}
    """
    print(f"Processing file: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    dia_id_col = df['Dialogue_ID'].tolist()
    sentiment_col = df['Sentiment'].tolist()
    emotion_col = df['Emotion'].tolist()

    all_emotions = []
    all_sentiments = []
    
    current_emotion_utt = []
    current_sentiment_utt = []

    # Iterate through the rows to group utterances into dialogues
    for i in range(len(dia_id_col)):
        current_emotion_utt.append(EMOTION_TO_ID[emotion_col[i]])
        current_sentiment_utt.append(SENTIMENT_TO_ID[sentiment_col[i]])

        # Check if it's the last utterance or if the next dialogue ID is different
        is_last_utterance = (i == len(dia_id_col) - 1)
        if not is_last_utterance and dia_id_col[i] != dia_id_col[i+1]:
            # End of the current dialogue, so save the collected labels
            all_emotions.append(current_emotion_utt)
            all_sentiments.append(current_sentiment_utt)
            # Reset for the next dialogue
            current_emotion_utt = []
            current_sentiment_utt = []
    
    if current_emotion_utt:
        all_emotions.append(current_emotion_utt)
        all_sentiments.append(current_sentiment_utt)

    print(f"Found {len(all_emotions)} dialogues.")
    return {"emotion": all_emotions, "sentiment": all_sentiments}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MELD dataset CSV files to extract labels.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing the MELD CSV files (e.g., 'train_sent_emo.csv')."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="./emotion_sentiment_labels.pkl",
        help="Path to save the output pickle file."
    )
    args = parser.parse_args()

    files_to_process = {
        "train": "train_sent_emo.csv",
        "dev": "dev_sent_emo.csv",
        "test": "test_sent_emo.csv"
    }

    final_results = {}

    for split_name, filename in files_to_process.items():
        full_path = os.path.join(args.data_dir, filename)
        
        result = process_dialogues_from_csv(full_path)
        if result:
            final_results[split_name] = result

    if final_results:
        print(f"Saving combined results to {args.output_file}...")
        with open(args.output_file, 'wb') as f:
            pkl.dump(final_results, f)
        print("Done.")
    else:
        print("No data was processed, output file will not be created.")