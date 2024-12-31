import streamlit as st
from app.utils.simplifier import ScriptGenerator
from app.utils.voice_generator import PodcastGenerator
from pipeline import Pipeline
import os
import logging
import time
import shutil
st.set_page_config(layout="wide")
script = ScriptGenerator()
pipeline = Pipeline()
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Caching pipeline and text extraction
@st.cache_data
def extract_text(uploaded_file):
    return pipeline.common(uploaded_file)

# Caching podcast scripts generation
@st.cache_data
def generate_podcast_scripts(structured_text):
    return script.generate_podcast_episodes(structured_text)

# Caching podcast audio generation
@st.cache_data
def generate_podcast_audio(scripts, model):
    podcast = PodcastGenerator(model)
    return podcast.process_scripts(scripts)

OUTPUT_FOLDER = "./outputs"

def clear_output_folder():
    """Deletes all files in the output folder when the app is refreshed."""
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)

clear_output_folder()

def main():
    try:
        st.title("AudioThesis")
        st.write("No more reading")
        uploaded_file = st.file_uploader("Upload a file", type=["pdf"])

        if uploaded_file:
            if uploaded_file.name.endswith(".pdf"):
                
                structured_text = extract_text(uploaded_file)

                choice = st.sidebar.radio("Choose One", ("Podcast Episodes Generate", "Question N Answer"))

                if choice == "Podcast Episodes Generate":
                    process_podcast(structured_text)
                else:
                    process_qa(structured_text, uploaded_file)
            else:
                st.info("Please upload a valid document in PDF format.")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        st.error("An unexpected error occurred. Please contact support.")

def process_podcast(structured_text):
    try:
        author = st.radio("Select the Voice of the Author", ("Male", "Female"))
        model = "aura-venus-en" if author == "Female" else "aura-helios-en"

        if st.button("Generate"):
            scripts = generate_podcast_scripts(structured_text)
            logger.info("Scripts generated successfully.")

            podcast_audio = generate_podcast_audio(scripts, model)
            #output_folder = 
            audio_files = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".wav")]

            if audio_files:
                st.write("Available Audio Episodes:")
                for audio_file in audio_files:
                    display_name = audio_file.replace("_", " ").replace(".wav", "")
                    st.write(display_name)
                    audio_file_path = os.path.join(OUTPUT_FOLDER, audio_file)
                    st.audio(audio_file_path, format="audio/wav")
                    with open(audio_file_path, "rb") as audio_file_data:
                        
                        st.download_button(
                            label=f"Download",
                            data=audio_file_data,
                            file_name=audio_file,
                            mime="audio/wav"
                        )
            else:
                st.write("No audio files available. Please try again after generating audio.")
    except Exception as e:
        logger.error(f"Error in podcast generation: {e}")
        st.error("Failed to generate podcast. Please try again.")

def process_qa(structured_text, uploaded_file):
    try:
        
        if "agent" not in st.session_state:
            st.session_state.agent = pipeline.answer_generate(structured_text, uploaded_file)

        st.title("Ask your questions from the research paper")
        question = st.text_input("Enter your question:")
       

        if st.button("Answer"):
            s = time.time()
            agent = st.session_state.agent
            answer = agent.get_answer(question)
            answer = agent.response(answer)
            st.write(answer)
            logger.info(f"Time taken: {time.time() - s:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in QA processing: {e}")
        st.error("Failed to process your question. Please try again.")

if __name__ == "__main__":
    main()
