# import os
# import pyttsx3
# from TTS.api import TTS

# class AudioGenerator:
#     def __init__(self):
#         self.tts_model = TTS('tts_models/en/ljspeech/tacotron2-DDC', gpu=False)
    
#     def generate_audio(self, text: str, voice: str, audio_format: str = "mp3") -> str:
#         output_path = f"output/audio/podcast.{audio_format}"
        
#         # Generate audio using the selected voice
#         if voice == "es":
#             self.tts_model.tts_to_file(text, speaker="spanish_female", file_path=output_path)
#         else:
#             self.tts_model.tts_to_file(text, file_path=output_path)
        
#         return output_path
