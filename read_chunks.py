import requests
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_embedding(text_list):
    r =requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3", 
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding

jsons = os.listdir("jsons")
my_dict = []
chunk_id = 0
for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    embeddings = create_embedding([c["text"] for c in content["chunks"]])
    print(f"Creating Embeddings for {json_file}")
    for i, chunk in enumerate(content["chunks"]):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id += 1
        my_dict.append(chunk)
    break


df = pd.DataFrame.from_records(my_dict)

input_query = input("Ask a question: ")
question_embedding = create_embedding([input_query])[0]

# Find similarities between question_embedding and input_query
# print(np.vstack(df["embedding"].values)) #Converting Embeggings values into 2d numpy array
# print(np.vstack(df["embedding"]).shape) #Cosine similarity funtion onyl takes 2d array as input 
similarities = cosine_similarity(np.vstack(df["embedding"]), [question_embedding]).flatten()

top_results = 4
max_indx = similarities.argsort()[::-1][0:top_results]
new_df = df.loc[max_indx]
print(new_df[["title", "number", "text"]])