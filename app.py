import streamlit as st
import os
import subprocess
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from process_incoming import create_embedding, inference
import json
import re
import threading
import time

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="VidQuery AI - Video RAG Assistant",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_processing_state():
    return {
        "is_processing": False,
        "status": "",
        "logs": "",
        "error": "",
        "completed": False
    }

processing_state = get_processing_state()

# Custom premium styling

# Custom premium styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

/* Main font overrides */
html, body, [data-testid="stAppViewContainer"], .main {
    font-family: 'Outfit', sans-serif;
    background-color: #0d0f14;
    color: #f1f3f9;
}

/* Title Styling */
.title-text {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60efff 0%, #0061ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.subtitle-text {
    font-size: 1.1rem;
    color: #8b9bb4;
    margin-bottom: 2rem;
}

/* Glassmorphism Sidebar styling */
[data-testid="stSidebar"] {
    background-color: rgba(17, 22, 34, 0.95) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Premium Card container */
.css-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}

/* Custom file uploader adjustments */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(255, 255, 255, 0.15);
    border-radius: 8px;
    padding: 10px;
    background-color: rgba(255, 255, 255, 0.01);
}

/* Status Indicator style */
.status-pill {
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 500;
    display: inline-block;
}
.status-ready {
    background-color: rgba(46, 213, 115, 0.15);
    color: #2ed573;
    border: 1px solid rgba(46, 213, 115, 0.3);
}
.status-empty {
    background-color: rgba(255, 71, 87, 0.15);
    color: #ff4757;
    border: 1px solid rgba(255, 71, 87, 0.3);
}
</style>
""", unsafe_allow_html=True)

# Ensure folders exist
os.makedirs("videos", exist_ok=True)
os.makedirs("Audios", exist_ok=True)
os.makedirs("jsons", exist_ok=True)

# --- 2. Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

def get_db_mtime():
    if os.path.exists("embeddings.joblib"):
        return os.path.getmtime("embeddings.joblib")
    return 0

def delete_video_data(video_file):
    num_match = re.search(r'#(\d+)', video_file)
    if num_match:
        tutorial_number = num_match.group(1)
    else:
        starts_with_num = re.match(r'^(\d+)[_-]', video_file)
        tutorial_number = starts_with_num.group(1) if starts_with_num else "unknown"

    if " | " in video_file:
        file_name = video_file.split(" | ")[0].strip()
    else:
        file_name = os.path.splitext(video_file)[0].strip()

    video_path = os.path.join("videos", video_file)
    audio_path = os.path.join("Audios", f"{tutorial_number}_{file_name}.mp3")
    json_path = os.path.join("jsons", f"{tutorial_number}_{file_name}.json")

    for path in [video_path, audio_path, json_path]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Failed to remove {path}: {e}")

    if os.path.exists("embeddings.joblib"):
        try:
            df = joblib.load("embeddings.joblib")
            df_filtered = df[(df["number"] != str(tutorial_number)) | (df["title"] != file_name)]
            
            if len(df_filtered) > 0:
                df_filtered = df_filtered.copy()
                df_filtered["chunk_id"] = range(len(df_filtered))
                joblib.dump(df_filtered, "embeddings.joblib")
                st.session_state.df = df_filtered
                st.session_state.df_mtime = os.path.getmtime("embeddings.joblib")
            else:
                os.remove("embeddings.joblib")
                st.session_state.df = None
                st.session_state.df_mtime = 0
        except Exception as e:
            print(f"Error updating database during deletion: {e}")

db_mtime = get_db_mtime()

# Load/check embedding database dynamically
if "df" not in st.session_state or st.session_state.get("df_mtime", 0) < db_mtime:
    if os.path.exists("embeddings.joblib"):
        try:
            st.session_state.df = joblib.load("embeddings.joblib")
            st.session_state.df_mtime = db_mtime
        except Exception as e:
            st.session_state.df = None
            st.session_state.df_mtime = 0
    else:
        st.session_state.df = None
        st.session_state.df_mtime = 0

# --- 3. Sidebar UI (Video Upload & Library Player) ---
with st.sidebar:
    st.markdown("<h2 style='font-family:\"Space Grotesk\", sans-serif; color: #f1f3f9; margin-top: 0;'>📂 Media Manager</h2>", unsafe_allow_html=True)
    
    # Database Status Pill
    if st.session_state.df is not None:
        st.markdown(f'<div class="status-pill status-ready">🟢 Database Ready ({len(st.session_state.df)} Chunks)</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-pill status-empty">🔴 Database Empty (Please Index Videos)</div>', unsafe_allow_html=True)
    
    st.write("---")
    
    # 3.1 Drag-and-Drop Uploader
    st.markdown("### 📤 Upload New Videos")
    uploaded_files = st.file_uploader("Choose video files", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join("videos", uploaded_file.name)
            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.toast(f"Saved: {uploaded_file.name}", icon="📁")
    
    # 3.2 Pipeline Automation trigger
    st.markdown("### ⚙️ Pipeline Control")
    
    if st.button("🚀 Process & Index All Videos", use_container_width=True, disabled=processing_state["is_processing"]):
        if not os.listdir("videos"):
            st.sidebar.error("No videos found in the 'videos/' folder. Please upload a video first!")
        else:
            processing_state["completed"] = False
            processing_state["error"] = ""
            processing_state["logs"] = ""
            
            def process_pipeline_worker():
                processing_state["is_processing"] = True
                try:
                    processing_state["status"] = "Step 1/3: Converting videos to MP3 audio..."
                    r1 = subprocess.run(["python3", "videos_to_mp3s.py"], capture_output=True, text=True, check=True)
                    processing_state["logs"] += r1.stdout + "\n"
                    
                    processing_state["status"] = "Step 2/3: Transcribing audio files (Faster-Whisper)..."
                    r2 = subprocess.run(["python3", "mp3_to_json.py"], capture_output=True, text=True, check=True)
                    processing_state["logs"] += r2.stdout + "\n"
                    
                    processing_state["status"] = "Step 3/3: Generating embeddings & building database..."
                    r3 = subprocess.run(["python3", "preprocess_jsons.py"], capture_output=True, text=True, check=True)
                    processing_state["logs"] += r3.stdout + "\n"
                    
                    processing_state["status"] = "✅ Indexing successfully completed!"
                    processing_state["completed"] = True
                except subprocess.CalledProcessError as e:
                    processing_state["error"] = f"Pipeline failed during {processing_state['status']} with exit code {e.returncode}"
                except Exception as e:
                    processing_state["error"] = f"Pipeline failed: {e}"
                finally:
                    processing_state["is_processing"] = False

            # Start background thread
            thread = threading.Thread(target=process_pipeline_worker)
            thread.start()
            st.rerun()

    # Display real-time UI for processing
    if processing_state["is_processing"] or processing_state["error"] or processing_state["completed"]:
        st.markdown("#### Processing Status")
        status_box = st.empty()
        
        if processing_state["error"]:
            status_box.error(processing_state["error"])
            if st.button("Dismiss Error"):
                processing_state["error"] = ""
                st.rerun()
        elif processing_state["completed"]:
            status_box.success(processing_state["status"])
            with st.expander("Show processing details", expanded=False):
                st.code(processing_state["logs"])
            
            # Reload dataframe safely in main thread after completion
            try:
                if os.path.exists("embeddings.joblib"):
                    st.session_state.df = joblib.load("embeddings.joblib")
                    st.session_state.df_mtime = os.path.getmtime("embeddings.joblib")
            except Exception as load_e:
                print(f"Error reloading DB after process: {load_e}")

            if st.button("Dismiss Status"):
                processing_state["completed"] = False
                processing_state["logs"] = ""
                st.rerun()
        else:
            status_box.info(processing_state["status"] + " ⏳")
            with st.expander("Live Logs", expanded=True):
                st.code(processing_state["logs"] or "Initializing...")
            # Auto-refresh loop while processing
            time.sleep(1)
            st.rerun()

    st.write("---")
    
    # 3.3 Media Player in Sidebar
    st.markdown("### 📺 Video Library")
    video_files = [f for f in os.listdir("videos") if f.endswith((".mp4", ".mov", ".avi", ".mkv"))]
    
    # Sort files by their tutorial number if possible
    def get_tutorial_num(filename):
        match = re.search(r'#(\d+)', filename)
        return int(match.group(1)) if match else 999
    video_files.sort(key=get_tutorial_num)

    def format_video_name(filename):
        num_match = re.search(r'#(\d+)', filename)
        if num_match:
            tutorial_number = num_match.group(1)
        else:
            starts_with_num = re.match(r'^(\d+)[_-]', filename)
            tutorial_number = starts_with_num.group(1) if starts_with_num else "?"
            
        if " | " in filename:
            clean_name = filename.split(" | ")[0].strip()
        else:
            clean_name = os.path.splitext(filename)[0].strip()
            
        return f"[{tutorial_number}] {clean_name}"

    if video_files:
        selected_video = st.selectbox("Select a video to play", video_files, format_func=format_video_name)
        if selected_video:
            st.video(os.path.join("videos", selected_video))
            
            with st.expander("⚠️ Danger Zone"):
                st.warning(f"Are you sure you want to delete **{selected_video}**? This will permanently remove the video, audio, transcripts, and database embeddings.")
                if st.button("🗑️ Delete Video", type="primary", use_container_width=True):
                    with st.spinner("Deleting video and cleaning database..."):
                        delete_video_data(selected_video)
                        st.toast("Video deleted successfully!", icon="✅")
                        st.rerun()
    else:
        st.info("No videos processed yet. Drag a video file above!")

# --- 4. Main Chat Interface ---
st.markdown('<div class="title-text">🎧 VidQuery AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Interactive RAG chatbot powered by Faster-Whisper, Scikit-Learn, and Groq LLM</div>', unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt input
if query := st.chat_input("Ask a question about the video content..."):
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
        
    # Generate RAG response
    with st.chat_message("assistant"):
        if st.session_state.df is None:
            err_msg = "The vector database is currently empty. Please upload at least one video and run the **Process & Index** pipeline in the sidebar!"
            st.error(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})
        else:
            response_container = st.empty()
            with st.spinner("Searching video context & generating answer..."):
                try:
                    # Double-check database sync right before RAG query to avoid any OS/Streamlit state sync lag
                    current_db_mtime = get_db_mtime()
                    if st.session_state.get("df_mtime", 0) < current_db_mtime:
                        if os.path.exists("embeddings.joblib"):
                            try:
                                st.session_state.df = joblib.load("embeddings.joblib")
                                st.session_state.df_mtime = current_db_mtime
                            except Exception as le:
                                print("Error in dynamic safety reload:", le)
                                
                    df = st.session_state.df
                    
                    # 1. Embed query
                    question_embedding = create_embedding([query])[0]
                    
                    # 2. Compute Cosine Similarity
                    similarities = cosine_similarity(np.vstack(df["embedding"]), [question_embedding]).flatten()
                    
                    # 3. Fetch top results (Context)
                    top_results = 8
                    max_indx = similarities.argsort()[::-1][0:top_results]
                    new_df = df.iloc[max_indx]
                    
                    # 4. Formulate Prompt
                    context_list = []
                    for _, row in new_df.iterrows():
                        def format_time(seconds):
                            m = int(seconds // 60)
                            s = int(seconds % 60)
                            return f"{m:02d}:{s:02d}"
                        
                        context_list.append({
                            "title": row["title"],
                            "number": row["number"],
                            "start": format_time(row["start"]),
                            "end": format_time(row["end"]),
                            "text": row["text"]
                        })
                    context_json = json.dumps(context_list, indent=2)
                    
                    prompt = f"""
You are an AI assistant that answers questions based on the provided video content.

Context:
{context_json}

------------------------------------------------

User Question:
{query}

Instructions:
- Answer clearly and naturally (like a teacher explaining).
- Mention relevant video number(s) and timestamps.
- Use 2–4 most relevant timestamps (avoid too many).
- Keep explanation helpful and easy to understand.

- If the question is unrelated, say:
  "This question is not related to the available video content."

Do NOT:
- Guess timestamps that are not present
- Mention anything like "not available" or "based on context"

Give a clean, human-like answer.
"""
                    
                    # 5. Fetch LLM response
                    response, provider = inference(prompt)
                    
                    response_container.markdown(response)
                    
                    # Store response in chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    err_msg = f"An error occurred while generating the answer: {e}"
                    response_container.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
