# 🎧 VidQuery AI (RAG Pipeline)

## 📌 Project Overview
VidQuery AI is an end-to-end Retrieval-Augmented Generation (RAG) system that converts video content into a searchable AI assistant. It processes videos into text, generates embeddings, and allows users to ask questions and receive precise answers with relevant video timestamps.

---

## 🚀 Features
- 🎥 Convert videos to audio (MP3)
- 📝 Transcribe audio using Faster-Whisper (optimized in-memory grouping)
- ✂️ Smart chunk grouping for better context
- 🔍 Generate embeddings for semantic search
- 🤖 Retrieve relevant chunks using cosine similarity
- 💬 Answer queries using LLM (Groq + Ollama fallback)
- ⏱️ Provides video number + accurate timestamps
- 🖥️ **Streamlit Web UI**: Premium dark mode dashboard with drag-and-drop file uploader, progress spinner, interactive chat interface, and sidebar video player.
- 🗑️ **Media Management**: Easily delete processed videos along with their audio, transcripts, and database embeddings directly from the UI.

---

## 📂 Project Structure
```
project/
│
├── videos/                 # Input folder for videos
├── Audios/                 # Extracted audio files
├── jsons/                  # Generated structured transcripts
├── embeddings.joblib       # Vector database
│
├── app.py                  # Streamlit Web Interface (Main Entry Point)
├── videos_to_mp3s.py       # Converts videos to MP3
├── mp3_to_json.py          # Transcribes audio to chunks
├── preprocess_jsons.py     # Generates database embeddings
├── process_incoming.py     # CLI Query interface (Optional)
└── .env                    # Environment credentials (Git-ignored)
```

---

## ⚙️ Running the Project

### 1️⃣ Option A: Running the Web Application (Recommended)
You can upload, process, watch, and chat with all your videos directly through a beautiful web browser interface:
```bash
streamlit run app.py
```
*Your browser will automatically open `http://localhost:8501`.*

---

### 2️⃣ Option B: Running via CLI Scripts (Terminal)

#### Step 1: Convert Videos to MP3
```bash
python3 videos_to_mp3s.py
```

#### Step 2: Transcribe & Group Chunks
```bash
python3 mp3_to_json.py
```

#### Step 3: Generate Embeddings
```bash
python3 preprocess_jsons.py
```

#### Step 4: Ask Questions
```bash
python3 process_incoming.py
```

---

## 🧠 Tech Stack
- Python
- Faster-Whisper (base model, INT8 quantized)
- Ollama
- Groq API
- Pandas / NumPy
- Scikit-learn
- Streamlit
- FFmpeg

---

## 📦 Installation & Setup

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn joblib requests faster-whisper groq python-dotenv streamlit
```

### 2. Install FFmpeg
On macOS:
```bash
brew install ffmpeg
```

### 3. Setup Groq API Key
Create a `.env` file in the root of the project and add your key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Setup Local Ollama Models
Ensure Ollama is running and download the embedding and fallback models:
```bash
ollama run llama3.2
ollama pull bge-m3
```

---

## 💡 How It Works
1. Video → audio  
2. Audio → text  
3. Text → chunks  
4. Embeddings  
5. Query → cosine similarity search  
6. LLM generates answer  

---

## 👨‍💻 Author
Shlok Bhandari  
B.E. Artificial Intelligence & Data Science

