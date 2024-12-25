from fastapi import File, APIRouter, UploadFile, HTTPException, Query
from app.utils import file_handler
from fastapi.responses import FileResponse
from app.utils.text_extractor import ExtractionAndChunking
from app.utils.summarizer import GeminiSummarizer
from app.utils.simplifier import ScriptGenerator
from app.utils.voice_generator import PodcastGenerator

router = APIRouter()

#sumarizer_handler = GeminiSummarizer()
script = ScriptGenerator()

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...),model: str = Query("aura-helios-en", enum=["aura-helios-en", "aura-luna-en"])):
                        
    if not file.filename.endswith(('.pdf' , '.docx')):
        raise HTTPException(status_code=400 , detail = "Invalid file format. Please upload file only in PDF or DOCX extensions")
    
    try:
        file_path = await file_handler.save(file)
        extractor = ExtractionAndChunking(file_path)
        extracted_text = extractor.extract_text_from_pdf_pymupdf()

        structured_text = extractor.model(extracted_text)
        scripts = script.generate_podcast_episodes(structured_text)
        # summarized_text = sumarizer_handler.summarize(structured_text)
        podcast = PodcastGenerator(model)
        podcast_audio = podcast.process_scripts(scripts)
        # sumarizer_handler.save_summarized_output(summarized_text)

        return {
            "extracted_text": extracted_text,
            "structured_text": structured_text,
            "Scripts": scripts,
            "podcast_audio": podcast_audio
            #"summarized_text": summarized_text,
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
