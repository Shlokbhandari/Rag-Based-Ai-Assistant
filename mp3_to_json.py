from faster_whisper import WhisperModel
import json
import os
import math
import shutil

model = WhisperModel("base", compute_type="int8")

audios = os.listdir("Audios")

for audio in audios: 
    if "_" in audio:
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
        print(number, title)

        segments, info = model.transcribe(
            f"Audios/{audio}",
            language="hi",
            task="translate"
        )

        chunks = []
        for segment in segments:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })

        full_text = " ".join([seg.text for seg in segments])
        chunks_with_metadata = {"chunks": chunks, "text": full_text}

        with open(f"jsons/{audio}.json", "w") as f:
            json.dump(chunks_with_metadata, f)

n = 5

jsons = os.listdir("jsons")

for filename in jsons:
    if filename.endswith(".json"):
        file_path = os.path.join("jsons", filename)
        with open (file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            new_chunks = []
            num_chunks = len(data["chunks"])
            num_groups = math.ceil(num_chunks/n)
        
        for i in range(num_groups):
            st_idx = i*n
            end_idx = min((i+1)*n, num_chunks)

            chunk_group = data["chunks"][st_idx: end_idx]
            new_chunks.append({
                "number" : data["chunks"][0]["number"],
                "title" : chunk_group[0]["title"],
                "start" : chunk_group[0]["start"],
                "end" : chunk_group[-1]["end"],
                "text" : " ".join(c["text"] for c in chunk_group)
            })

        #Save the file without double .json name
        os.makedirs("newjsons", exist_ok=True)
        with open (os.path.join("newjsons", filename), "w", encoding="utf-8") as json_file:
            json.dump({"chunks": new_chunks, "text": data["text"]}, json_file, indent=4)
            
shutil.rmtree("jsons")

print("✅ Processing complete.")