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

def inference(prompt):
    r =requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2", 
        "prompt": prompt,
        "stream": False
    })

    response =r.json()
    return response

df = joblib.load("embeddings.joblib")
input_query = input("Ask a question: ")
question_embedding = create_embedding([input_query])[0]

# Find similarities between question_embedding and input_query
# print(np.vstack(df["embedding"].values)) #Converting Embeggings values into 2d numpy array
# print(np.vstack(df["embedding"]).shape) #Cosine similarity funtion onyl takes 2d array as input 
similarities = cosine_similarity(np.vstack(df["embedding"]), [question_embedding]).flatten()

top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
new_df = df.loc[max_indx]

prompt = f'''
I am teaching web development using Sigma Web Development couse. Here are video chunks with video title, video number, start seconds of the chunk. end seconds of the chunk , and the text at that time:

{new_df[["title", "number", "start", "end", "text"]].to_json(orient = "records")}
------------------------------------------------
{input_query}
User aked this question related to video chunks, you have to answer in a human way (dont mention the above format, ts just for you) where and how much content is taught in which video (in which video and what timstamp) and guide the user to go to that particular video. If user asks unrelated questions, tell him that you can only answer questions related to the course.
'''

response = inference(prompt)["response"]
print(response)
with open ("response.txt", "w") as f:
    f.write(response)