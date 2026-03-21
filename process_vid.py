#Convert Videos to Mp3
import os
import subprocess

input_folder = "Videos"
output_folder = "Audios"

files = [f for f in os.listdir(input_folder) if f.endswith(".mp4") and not f.startswith(".")]

for file in files:
    tutorial_number = file.split("_")[0]
    file_name = file.split("_")[1]

    input_path = os.path.join(input_folder, file)
    output_path = os.path.join(output_folder, f"{tutorial_number}_{file_name}.mp3")
    subprocess.run(["ffmpeg", "-i", input_path, output_path])