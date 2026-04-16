# 🎧 VidQuery AI (RAG Pipeline)

## 📌 Project Overview
This project builds an end-to-end pipeline that converts video content into a searchable AI assistant using speech-to-text transcription, text chunking, embeddings, semantic search, and LLM-based answer generation.

It allows users to ask questions about video content and get precise answers along with video references and timestamps.

---

## 🚀 Features
- 🎥 Convert videos to audio (MP3)
- 📝 Transcribe audio using Whisper
- 🔍 Generate embeddings for semantic search
- 🤖 Retrieve relevant chunks using cosine similarity
- 💬 Answer queries using LLM (RAG approach)
- ⏱️ Provides video number + timestamps

---

## 📂 Project Structure
project/
│
├── videos/ # Input videos
├── audios/ # Extracted audio files
├── jsons/ # Transcribed chunks
├── embeddings.joblib # Stored embeddings
│
├── vid_to_mp3.py # Convert videos → mp3
├── mp3_to_jsons.py # Transcribe audio → JSON
├── preprocess_json.py # Create embeddings
├── process_incoming.py # Query + answer system


---

## ⚙️ Workflow

### 1️⃣ Convert Videos to MP3
Uses ffmpeg to extract audio from videos:

python vid_to_mp3.py


### 2️⃣ Transcribe Audio
Uses Whisper (large-v2 model) to convert audio into text chunks with timestamps:

python mp3_to_jsons.py


### 3️⃣ Generate Embeddings
Uses bge-m3 embedding model via Ollama and stores them in a file:

python preprocess_json.py


### 4️⃣ Ask Questions (Inference)
Takes user query, retrieves relevant chunks, and generates answers:

python process_incoming.py


---

## 🧠 Tech Stack
- Python
- Whisper (Speech-to-Text)
- Ollama (LLM + Embeddings)
- Pandas / NumPy
- Scikit-learn (Cosine Similarity)
- FFmpeg

---

## 📦 Requirements
Install dependencies:

pip install pandas numpy scikit-learn joblib requests openai-whisper


Install FFmpeg (Mac):

brew install ffmpeg


---

## ⚠️ Prerequisites
Run Ollama locally:

ollama run llama3.2
ollama pull bge-m3


Ensure APIs are running:

http://localhost:11434/api/embed

http://localhost:11434/api/generate


---

## 💡 How It Works (RAG Pipeline)
1. Convert video → audio  
2. Audio → text (Whisper)  
3. Text → chunks + embeddings  
4. User query → embedding  
5. Compare with stored embeddings  
6. Retrieve top relevant chunks  
7. Send to LLM → generate answer  

---

## 📊 Example
Input:

Ask a question: What is HTML?


Output:
- Suggests relevant video  
- Shows timestamps  
- Explains concept clearly  

---

## 🎯 Applications
- Online course assistants  
- Lecture search systems  
- YouTube content Q&A  
- Educational AI tools  

---

## 🔮 Future Improvements
- Add web UI (Streamlit / React)
- Use vector database (FAISS / Pinecone)
- Multi-language support
- Real-time query system

---

## 👨‍💻 Author
Shlok Bhandari  
B.E. Artificial Intelligence & Data Science
