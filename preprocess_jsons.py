import requests
import os
import json
import pandas as pd
import numpy as np
import joblib

def create_embedding(text_list):
    r =requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3", 
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

jsons = [f for f in os.listdir("jsons") if f.endswith(".json")]

# Load existing database to check for already processed videos
existing_df = None
processed_videos = set()
chunk_id = 0

if os.path.exists("embeddings.joblib"):
    try:
        existing_df = joblib.load("embeddings.joblib")
        if "number" in existing_df.columns and "title" in existing_df.columns:
            # Create a set of (number, title) to quickly check existence
            processed_videos = set(zip(existing_df["number"].astype(str), existing_df["title"]))
            # Start new chunk IDs after the existing 1's
            if len(existing_df) > 0:
                chunk_id = int(existing_df["chunk_id"].max()) + 1
    except Exception as e:
        print(f"Warning: Could not load existing database: {e}")

my_dict = []
for json_file in jsons:
    with open(f"jsons/{json_file}", "r", encoding="utf-8") as f:
        content = json.load(f)
        
    if not content.get("chunks"):
        continue
        
    # Get metadata from the first chunk to check if we already processed it
    first_chunk = content["chunks"][0]
    vid_num = str(first_chunk.get("number", "unknown"))
    vid_title = first_chunk.get("title", "")
    
    if (vid_num, vid_title) in processed_videos:
        print(f"Skipping '{json_file}' (already embedded)")
        continue

    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c["text"] for c in content["chunks"]])
    
    for i, chunk in enumerate(content["chunks"]):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        my_dict.append(chunk)

# Combine and save
if len(my_dict) > 0:
    new_df = pd.DataFrame.from_records(my_dict)
    if existing_df is not None and len(existing_df) > 0:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Ensure chunk_id is perfectly sequential
        final_df["chunk_id"] = range(len(final_df))
    else:
        final_df = new_df
        final_df["chunk_id"] = range(len(final_df))
        
    joblib.dump(final_df, "embeddings.joblib")
    print(f"✅ Added {len(new_df)} new chunks to database.")
else:
    print("✅ Database is already up to date. No new chunks added.")