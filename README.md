# 🎙️ Multi-Speaker Audio Transcription and Speaker Diarization System

This project focuses on developing an advanced pipeline for **transcribing multi-speaker audio** and accurately performing **speaker diarization**. It integrates **OpenAI's Whisper model** for high-quality transcription and **SpeechBrain's ECAPA-TDNN** for extracting speaker embeddings, enabling precise speaker segmentation and labeling.

> 🔍 Built as part of an academic mini-project to explore real-world speech processing use cases like virtual meetings, interviews, and podcast analytics.

---

## 🎯 Objective

To design and implement a system that:
- 🎧 Transcribes multi-speaker audio files with high accuracy
- 🧠 Identifies and distinguishes speakers using speaker embeddings
- 🕒 Produces **timestamped**, **speaker-tagged** transcripts
- 📊 Visualizes speaker timelines for clear interpretation

---

## 🛠️ Technology Stack

| Component            | Tools / Libraries                        |
|----------------------|------------------------------------------|
| Speech Recognition   | OpenAI Whisper (Large/Medium model)      |
| Speaker Embedding    | SpeechBrain – ECAPA-TDNN                 |
| Clustering           | Agglomerative Clustering (Scikit-learn)  |
| Audio Processing     | Librosa, Pydub, NumPy                    |
| Visualization        | Matplotlib, seaborn, t-SNE               |
| Development Env      | Python, Jupyter Notebook                 |

---

## 🧩 System Architecture

### 🔄 Pipeline Steps:
1. **Audio Preprocessing**
   - Normalize sample rate
   - Convert to mono-channel
   - Segment into uniform-length chunks

2. **Transcription**
   - Pass segments through Whisper for accurate transcription
   - Extract timestamps for each spoken chunk

3. **Speaker Embedding**
   - Extract ECAPA-TDNN embeddings from SpeechBrain
   - Compare each segment's embedding with known speakers (if available)

4. **Speaker Clustering & Identification**
   - Cluster voice segments using Agglomerative Clustering
   - Use **cosine similarity** to match clusters with known speaker embeddings (if available)

5. **Output Generation**
   - Structured **transcript with speaker labels**
   - **Timeline visualization** (who spoke when)
   - **Tabular breakdown** with timestamps and speaker names

---

## 📈 Output Formats

- 🧾 **CSV Table**: Timestamped transcripts with speaker labels
- 📊 **Speaker Timeline Chart**: Visual representation of speaker activity over time
- 🧠 **t-SNE Plots**: Visualization of speaker clusters in high-dimensional space

---

## 💻 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/multi-speaker-diarization
cd multi-speaker-diarization

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the diarization pipeline
python diarize.py --audio path/to/audio.wav --reference_dir path/to/reference_voices/
