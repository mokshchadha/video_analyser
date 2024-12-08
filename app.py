import streamlit as st
import os
import tempfile
import google.generativeai as genai
import whisper
from pydub import AudioSegment

# For Streamlit Cloud deployment, use secrets instead of .env
def get_api_key():
    # Try to get from secrets first (for deployment)
    try:
        return st.secrets["GOOGLE_API_KEY"]
    # Fallback to environment variable (for local development)
    except:
        return os.getenv("GOOGLE_API_KEY")

# Configure the Gemini model
genai.configure(api_key=get_api_key())
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Load Whisper model - cache it to avoid reloading
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

whisper_model = load_whisper_model()

def extract_audio(file_path, output_path):
    audio = AudioSegment.from_file(file_path)
    audio.export(output_path, format="wav")

def transcribe_audio(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def analyze_text(text, prompt_template):
    # Combine the custom prompt template with the transcription
    full_prompt = f"{prompt_template}\n\nTranscription:\n{text}"
    response = model.generate_content(full_prompt)
    return response.text

def process_file(uploaded_file, prompt_template):
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_audio:
        audio_path = tmp_audio.name

    try:
        if file_extension in ['.mp4', '.mov', '.avi']:
            with st.spinner("Extracting audio from video..."):
                extract_audio(file_path, audio_path)
        elif file_extension in ['.m4a', '.wav', '.mp3']:
            with st.spinner("Converting audio to WAV format..."):
                extract_audio(file_path, audio_path)
        else:
            st.error("Unsupported file format. Please upload a video or audio file.")
            return None, None

        with st.spinner("Transcribing audio..."):
            transcription = transcribe_audio(audio_path)

        with st.spinner("Analyzing content..."):
            analysis = analyze_text(transcription, prompt_template)

        return transcription, analysis

    finally:
        # Clean up temporary files
        if os.path.exists(file_path):
            os.unlink(file_path)
        if os.path.exists(audio_path):
            os.unlink(audio_path)

def get_download_link(text, filename):
    """Generate a download link for text content"""
    b64 = text.encode()
    return f'<a href="data:text/plain;charset=utf-8,{b64.decode()}" download="{filename}">Download {filename}</a>'

def main():
    st.set_page_config(
        page_title="Video/Audio Analyzer",
        page_icon="🎥",
        layout="wide"
    )

    st.title("🎥 Video/Audio Content Analyzer")
    
    # Create two columns for the layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose a video or audio file", type=["mp4", "mov", "avi", "m4a", "wav", "mp3"])
        
        # Default prompt template
        default_prompt = """You are an expert interviewer. Please analyze this content and provide:
1. Key points discussed
2. Overall tone and delivery
3. Areas of strength
4. Areas for improvement
5. Overall assessment"""
        
        prompt_template = st.text_area(
            "Customize your analysis prompt",
            value=default_prompt,
            height=200,
            help="Modify this prompt to get different types of analysis from your content."
        )

    with col2:
        st.info("""
        ### How to use:
        1. Upload your video/audio file
        2. Customize the analysis prompt if needed
        3. Click 'Process File'
        4. Download transcription and analysis
        """)

    if uploaded_file is not None:
        if st.button("Process File"):
            transcription, analysis = process_file(uploaded_file, prompt_template)
            
            if transcription and analysis:
                # Create tabs for different views
                tab1, tab2 = st.tabs(["Analysis", "Transcription"])
                
                with tab1:
                    st.markdown("### Analysis")
                    st.markdown(analysis)
                    st.markdown(get_download_link(analysis, 'analysis.txt'), unsafe_allow_html=True)
                
                with tab2:
                    st.markdown("### Transcription")
                    st.markdown(transcription)
                    st.markdown(get_download_link(transcription, 'transcription.txt'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()