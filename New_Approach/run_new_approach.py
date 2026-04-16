import pandas as pd
import numpy as np
from transformers import pipeline
from tqdm.auto import tqdm
import csv
import os

# 1. SETUP & MODEL LOADING
# device=-1 forces CPU usage to keep your 2GB iGPU safe
print("\n--- [1/3] Loading Advanced NLP Model (BART-Large-MNLI) ---")
print("This model is ~1.6GB. If it's the first run, it will download automatically.")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)

# 2. DEFINE SEMANTIC LABELS (The "Semantic Anchors")
sentiment_labels = [
    "angry frustrated complaint", 
    "neutral statement", 
    "happy grateful appreciation"
]

action_labels = [
    "asking for information or clarifying details",
    "providing a distinct solution or technical answer",
    "apologizing or offering affective empathy repair",
    "escalating to support team or email",
    "closing the conversation goodbye",
    "giving a proactive system checking update",
    "setting expectations or asking to wait"
]

def classify_text(text, candidate_labels):
    text = str(text).strip()[:400]
    if len(text) < 2: return 1 # Default to Neutral
    
    # Zero-Shot Inference
    result = classifier(text, candidate_labels, multi_label=False)
    top_label = result['labels'][0]
    return candidate_labels.index(top_label)

# 3. PROCESSING LOOP
TWITTER_PATH = r"D:\SEM_6\RL\Project\my_local_twitter\twcs\twcs.csv"
OUTPUT_FILE = "nlp_offline_dataset_twitter.csv"

# Adjust nrows=None if you want to run the full million-row dataset overnight
N_ROWS = 5000 

print(f"\n--- [2/3] Processing Dataset ({N_ROWS} rows) ---")
if not os.path.exists(TWITTER_PATH):
    print(f"ERROR: Could not find dataset at {TWITTER_PATH}")
    exit()

df_tw = pd.read_csv(TWITTER_PATH, nrows=N_ROWS)

# Pre-index for speed
outbound = df_tw[df_tw['inbound'] == False].dropna(subset=['in_response_to_tweet_id'])
outbound_dict = {row['in_response_to_tweet_id']: row for _, row in outbound.iterrows()}
inbound_dict = {row['in_response_to_tweet_id']: row for _, row in df_tw.iterrows()}

nlp_tuples = []
inbound_df = df_tw[df_tw['inbound'] == True]

for _, initial_tweet in tqdm(inbound_df.iterrows(), total=len(inbound_df), desc="Extracting Tuples"):
    tid = initial_tweet['tweet_id']
    
    if tid in outbound_dict:
        agent_reply = outbound_dict[tid]
        
        # Semantic Classifications
        state_idx = classify_text(initial_tweet['text'], sentiment_labels)
        action_idx = classify_text(agent_reply['text'], action_labels)
        
        if agent_reply['tweet_id'] in inbound_dict:
            customer_reply = inbound_dict[agent_reply['tweet_id']]
            next_state_idx = classify_text(customer_reply['text'], sentiment_labels)
            
            # Mathematical Reward Logic
            # Reward = +10 for Happy, -10 for Angry, -2 for neutral (staying the same)
            reward = 10.0 if next_state_idx == 2 else (-10.0 if next_state_idx == 0 else -2.0)
            
            nlp_tuples.append((state_idx, action_idx, reward, next_state_idx, True))

# 4. SAVE RESULTS
print(f"\n--- [3/3] Saving Results to {OUTPUT_FILE} ---")
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["state", "action", "reward", "next_state", "done"])
    writer.writerows(nlp_tuples)

print(f"\nSUCCESS: Saved {len(nlp_tuples)} transitions using BART classification.")
print("You can now use this CSV to retrain your RL Agent without keyword bias.")
