import streamlit as st
from app.utils.text_extractor import ExtractionAndChunking
from app.utils.summarizer import GeminiSummarizer
from app.utils.simplifier import ScriptGenerator
from app.utils.voice_generator import PodcastGenerator
from app.utils import file_handler
import os

st.set_page_config(layout="wide") 
script = ScriptGenerator()

def main():

    st.title("AudioThesis")
    st.write("No more reading")
    uploaded_file = st.file_uploader("Upload a file", type=["pdf"])

    if uploaded_file is not None:
            if uploaded_file.name.endswith('.pdf'):
                
                Author = st.radio("Select the Voice of the author" , ("Male", "Female"))

                if Author == "Female":
                    model = "aura-venus-en"
                else:
                    model = "aura-helios-en"
                
                if st.button("Generate"):
                    file_path = file_handler.save(uploaded_file)
                    extractor = ExtractionAndChunking(file_path)
                    extracted_text = extractor.extract_text_from_pdf_pymupdf()

                    structured_text = extractor.model(extracted_text)
                    print(structured_text)
                    scripts = script.generate_podcast_episodes(structured_text)
                    print(scripts)
                    podcast = PodcastGenerator(model)
                    podcast_audio = podcast.process_scripts(scripts)

                    output_folder = "/Audiothesis/outputs"
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                        
                    audio_files = [f for f in os.listdir(output_folder) if f.endswith(".mp3")]

                    if audio_files:
                        st.write("Available Audio Episodes:")

                        for audio_file in audio_files:
                            audio_file_path = os.path.join(output_folder, audio_file)

                            st.audio(audio_file_path, format='audio/mp3')

                            with open(audio_file_path, "rb") as audio_file_data:
                                st.download_button(
                                    label=f"Download {audio_file}",
                                    data=audio_file_data,
                                    file_name=audio_file,
                                    mime="audio/mp3"
                                )
                    else:
                        st.write("No audio files available. Please try again after generating audio.")

            else:
                st.info("Please Upload a valid Document in pdf format")


if __name__ == "__main__":
    main()

