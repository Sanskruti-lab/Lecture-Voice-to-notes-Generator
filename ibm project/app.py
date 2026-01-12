import streamlit as st
from transformers import pipeline
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)
import speech_recognition as sr
from pydub import AudioSegment
import re
import os
import io

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Lecture AI", page_icon="🎓", layout="wide")
st.title("🎓 AI Lecture Voice-to-Notes Generator")
st.markdown("Convert your spoken lectures or YouTube videos into structured notes and quizzes.")

# --- 2. API KEY SETUP ---
# No API key needed anymore - using free local AI model!
#st.info("ℹ️ Using free, local AI model for note generation (no credits needed!)")

# --- 3. HELPER FUNCTIONS ---
def extract_video_id(url):
    # This handles regular links AND youtu.be short links
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def transcribe_audio(audio_file):
    """Transcribe audio from ANY format including video files (MP4, AVI, MOV, etc.)."""
    try:
        import tempfile
        
        filename = audio_file.name if hasattr(audio_file, 'name') else 'audio.wav'
        file_ext = filename.split('.')[-1].lower()
        
        st.info(f"📁 Processing {filename}...")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
            tmp_file.write(audio_file.getbuffer())
            temp_path = tmp_file.name
        
        try:
            recognizer = sr.Recognizer()

            # Goal: normalize any input into a pydub.AudioSegment, then chunk and recognize
            audio = None

            # Video formats - extract audio first
            if file_ext.lower() in ['mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv', 'webm']:
                st.info(f"Extracting audio from {file_ext.upper()} video...")
                try:
                    from moviepy.video.io.VideoFileClip import VideoFileClip
                    video = VideoFileClip(temp_path)
                    audio_clip = video.audio
                    if audio_clip is None:
                        return "❌ No audio track found in video file"
                    # write temporary wav
                    wav_path = temp_path.replace(f'.{file_ext}', '_audio.wav')
                    audio_clip.write_audiofile(wav_path)
                    video.close()
                    audio = AudioSegment.from_file(wav_path)
                    try:
                        os.unlink(wav_path)
                    except:
                        pass
                except Exception as e:
                    return f"❌ Cannot extract audio from {file_ext.upper()}: {str(e)}"

            else:
                # Audio formats (including wav/flac/mp3/m4a/ogg)
                try:
                    audio = AudioSegment.from_file(temp_path)
                except Exception:
                    # Try common loaders
                    try:
                        if file_ext == 'mp3':
                            audio = AudioSegment.from_mp3(temp_path)
                        elif file_ext == 'wav':
                            audio = AudioSegment.from_wav(temp_path)
                        elif file_ext == 'ogg':
                            audio = AudioSegment.from_ogg(temp_path)
                        else:
                            audio = AudioSegment.from_file(temp_path, format=file_ext)
                    except Exception as e:
                        return f"❌ Cannot load audio file: {str(e)}"

            if audio is None:
                return "❌ Could not obtain audio for transcription."

            # Standardize: mono, 16kHz
            audio = audio.set_frame_rate(16000).set_channels(1)

            # Chunk and transcribe (30s per chunk)
            chunk_ms = 30_000
            texts = []
            max_chunks = 10  # limit to ~5 minutes to avoid API issues
            total_len = len(audio)
            num_chunks = (total_len + chunk_ms - 1) // chunk_ms
            for i in range(min(num_chunks, max_chunks)):
                start = i * chunk_ms
                end = min(start + chunk_ms, total_len)
                chunk = audio[start:end]

                wav_io = io.BytesIO()
                chunk.export(wav_io, format='wav')
                wav_io.seek(0)

                try:
                    with sr.AudioFile(wav_io) as source:
                        audio_data = recognizer.record(source)
                        part = recognizer.recognize_google(audio_data)
                        texts.append(part)
                except sr.RequestError as e:
                    return f"❌ recognition request failed: {str(e)}"
                except sr.UnknownValueError:
                    texts.append("[Unintelligible]")

            result = " ".join(texts)
            if num_chunks > max_chunks:
                result += " [Note: Transcript truncated - first {} seconds only]".format(max_chunks * (chunk_ms // 1000))
            return result
            
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except sr.UnknownValueError:
        return "❌ Could not understand audio. Try clear speech with minimal background noise."
    except sr.RequestError as e:
        return f"❌ Internet required. Error: {str(e)}"
    except Exception as e:
        return f"❌ Error: {str(e)}"

def generate_ai_notes(transcript):
    """Generates study notes using free local transformer models."""
    # Load the summarization pipeline (runs locally, completely free!)
    @st.cache_resource
    def load_summarizer():
        return pipeline("summarization", model="facebook/bart-large-cnn")
    
    summarizer = load_summarizer()
    
    # Truncate transcript if too long (max 1024 tokens for BART)
    words = transcript.split()
    if len(words) > 900:
        transcript = " ".join(words[:900])
    
    try:
        # Generate summary
        summary = summarizer(transcript, max_length=150, min_length=50, do_sample=False)
        summary_text = summary[0]['summary_text']
    except Exception as e:
        summary_text = "Summary could not be generated."
    
    # Generate bullet points from key phrases
    sentences = transcript.split('. ')
    key_points = [s.strip() + '.' for s in sentences[:5] if len(s.strip()) > 20]
    bullet_points = "\n".join([f"• {point}" for point in key_points])
    
    # Generate quiz questions (simple keyword-based)
    important_words = sorted(set(transcript.lower().split()), key=len, reverse=True)[:10]
    quiz_questions = f"""
## 📋 Test Your Knowledge

1. **What was the main topic discussed?**
   - a) {important_words[0].capitalize()}
   - b) {important_words[1].capitalize()}
   - c) General information
   - **Answer: a**

2. **Key concepts mentioned:**
   - a) {important_words[2].capitalize() if len(important_words) > 2 else 'Topic'}
   - b) {important_words[3].capitalize() if len(important_words) > 3 else 'Discussion'}
   - c) Both of the above
   - **Answer: c**

3. **True or False: The transcript covered important details.**
   - **Answer: True**
"""
    
    notes = f"""
## 📝 Study Notes

### Executive Summary
{summary_text}

### Key Points
{bullet_points}

{quiz_questions}
"""
    
    return notes

# --- 4. THE USER INTERFACE (TABS) ---
tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎥 Paste YouTube Link"])

transcript_text = ""

with tab1:
    st.subheader("🎙️ Upload Audio Lecture")
    
    # Allow ALL file types
    uploaded_file = st.file_uploader(
        "📂 Choose any file",
        type=None  # Allows ALL file types
    )
    
    if uploaded_file:
        st.info(f"✅ Selected: {uploaded_file.name}")
        
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        # Show audio player for supported formats
        if file_ext in ['mp3', 'wav', 'm4a', 'ogg']:
            st.audio(uploaded_file)
        
        # Inform user we'll attempt to process the selected file type
        st.info(f"Processing {file_ext.upper()} file; attempting automatic conversion/extraction if needed.")
        
        # Transcribe button
        if st.button("🎙️ Transcribe Audio", key="transcribe_audio"):
            with st.spinner("Transcribing audio (may take 1-2 minutes)..."):
                class AudioData:
                    def __init__(self, data):
                        self.data = data
                        self.name = uploaded_file.name
                    def getbuffer(self):
                        return self.data
                
                audio_obj = AudioData(uploaded_file.getbuffer())
                transcript_text = transcribe_audio(audio_obj)
                
                if transcript_text and "Error" not in transcript_text:
                    st.success("✅ Transcription complete!")
                    st.session_state.transcript_text = transcript_text
                    st.text_area("Transcript:", value=transcript_text, height=200)
                else:
                    st.error(f"{transcript_text}")
    else:
        st.info("👆 Click 'Browse files' to select a file")
        
with tab2:
    yt_url = st.text_input("YouTube Lecture URL (e.g., https://youtube.com/watch?v=...)")
    if yt_url and st.button("Generate from YouTube"):
        video_id = extract_video_id(yt_url)
        if video_id:
            try:
                with st.spinner("Fetching YouTube transcript..."):
                    # First try the simple helper which prefers manually uploaded captions
                    # Use the instance API for the installed package version
                    ytt = YouTubeTranscriptApi()
                    try:
                        # shortcut: try to fetch a transcript (tries 'en' by default)
                        data = ytt.fetch(video_id)
                    except (NoTranscriptFound, CouldNotRetrieveTranscript, TranscriptsDisabled):
                        # Fall back to listing available transcripts and try to fetch any (including generated)
                        transcripts = ytt.list(video_id)
                        data = None
                        for t in transcripts:
                            try:
                                data = t.fetch()
                                if data:
                                    break
                            except Exception:
                                continue
                        if data is None:
                            # re-raise to be handled below
                            raise

                    # Convert transcript items to text (handle FetchedTranscriptSnippet objects)
                    transcript_text = " ".join([item.text if hasattr(item, 'text') else item['text'] for item in data])
                    st.success("YouTube Transcript retrieved!")
            except TranscriptsDisabled:
                st.error("Transcripts are disabled for this video.")
            except VideoUnavailable:
                st.error("Video unavailable — check the URL or region restrictions.")
            except NoTranscriptFound:
                st.error("No transcript found for this video. It may not have captions (CC).")
            except Exception as e:
                st.error(f"Could not get transcript: {str(e)}")
        else:
            st.error("Invalid YouTube URL.")

# --- 5. FINAL OUTPUT DISPLAY ---
if transcript_text:
    with st.spinner("AI is brainstorming your study guide..."):
        try:
            final_notes = generate_ai_notes(transcript_text)
            
            st.divider()
            st.subheader("📝 Your AI-Generated Study Guide")
            st.markdown(final_notes)
            
            # Download button for the student
            st.download_button(label="📥 Download Notes as Text", 
                               data=final_notes, 
                               file_name="lecture_notes.txt")
        except Exception as e:
            st.error(f"⚠️ Error generating notes: {str(e)}")