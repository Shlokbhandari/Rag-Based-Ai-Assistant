import requests
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib


def create_embedding(text_list):
    r =requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3", 
        "input": text_list
    })

    embedding = r.json()["embeddings"]
    return embedding


df = joblib.load("embeddings.joblib")
input_query = input("Ask a question: ")
question_embedding = create_embedding([input_query])[0]

# Find similarities between question_embedding and input_query
# print(np.vstack(df["embedding"].values)) #Converting Embeggings values into 2d numpy array
# print(np.vstack(df["embedding"]).shape) #Cosine similarity funtion onyl takes 2d array as input 
similarities = cosine_similarity(np.vstack(df["embedding"]), [question_embedding]).flatten()

top_results = 30
max_indx = similarities.argsort()[::-1][0:top_results]
new_df = df.loc[max_indx]

for index, item in new_df.iterrows():
    print(index, item["title"], item["number"], item["text"], item["start"], item["end"])