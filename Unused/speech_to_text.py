#Mp3 audio files to text and traslation in English
import whisper
import json
model = whisper.load_model("large-v2")

result = model.transcribe(audio = "Audios/01_HTML CheatSheet for Beginners .mp3", language = "hindi", task = "translate")


chunks = []
for segment in result["segments"]:
    chunks.append({"start": segment["start"], "end": segment["end"], "text": segment["text"]})

with open ("output.json", "w") as f:
    json.dump(chunks, f)