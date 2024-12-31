import time
import requests
import shutil
import os
import re

class PodcastGenerator:
    DG_API_KEY = os.environ.get("DG_API_KEY")

    def __init__(self, model):
        self.MODEL_NAME = model  

    @staticmethod
    def is_installed(lib_name: str) -> bool:
        """Check if a required library or tool is installed."""
        lib = shutil.which(lib_name)
        return lib is not None

    def speak(self, text, output_file):
        """Convert text to speech and save the audio."""
        
        if not self.is_installed("ffmpeg"):
            raise ValueError("ffmpeg not found. Install it with `sudo apt install ffmpeg`")

        
        DEEPGRAM_URL = (
            f"https://api.deepgram.com/v1/speak?model={self.MODEL_NAME}&encoding=linear16&sample_rate=24000"
        )
        headers = {
            "Authorization": f"Token {self.DG_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "text": text  
        }

        start_time = time.time()  
        first_byte_time = None  

        
        with requests.post(DEEPGRAM_URL, stream=True, headers=headers, json=payload) as r:
            if r.status_code != 200:
                raise Exception(f"Error: {r.status_code} - {r.json()}")

            
            with open(output_file, "wb") as audio_file:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                      
                        if first_byte_time is None:
                            first_byte_time = time.time()
                            ttfb = int((first_byte_time - start_time) * 1000)  # TTFB in ms
                            print(f"TTS Time to First Byte (TTFB): {ttfb}ms\n")
                        audio_file.write(chunk)

        print(f"Audio content written to {output_file}.")

    def process_scripts(self, episodes):
        """Process a dictionary of episodes and generate audio files for each."""
        
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        for episode_title, script_content in episodes.items():
            
            sanitized_title = re.sub(r"[^\w\s-]", "", episode_title).replace(" ", "_")
            output_file = os.path.join(output_dir, f"{sanitized_title}.wav")
            
            print(f"Processing: {episode_title}")
            self.speak(script_content, output_file)
            print(f"Saved audio for '{episode_title}' to {output_file}.")

            