# from app.utils.text_extractor import ExtractionAndChunking
# from app.utils.summarizer import GeminiSummarizer
# from app.utils import file_handler
# from app.utils.summarizer import GeminiSummarizer
# from app.utils.Cohere_Embedding import GenerateEmbedings, context_generation
# from app.utils.images_tables_extract import ImageTable
# from app.utils.langchain_handler import QA

# sumarizer_handler = GeminiSummarizer()

# class Pipeline:

#     def __init__(self, file):
#         self.uploaded_file = file

#     def common(self):
        
#         file_path = file_handler.save(self.uploaded_file)
#         extractor = ExtractionAndChunking(file_path)
#         extracted_text = extractor.extract_text_from_pdf_pymupdf()
#         structured_text = extractor.model(extracted_text)

#         return structured_text
    
#     def QA(self,structured_text):

#         self.structured_text = structured_text
#         summarized_text = sumarizer_handler.summarize(structured_text)

#         image_table = ImageTable(self.uploaded_file)
#         images, tables = image_table.extract_images_and_tables_from_pdf()

#         context = context_generation(summarized_text)

#         image_summary = sumarizer_handler.Image_summarize(context, images)
#         tables_summary = sumarizer_handler.Table_summarize(context, tables)
        
#         embeddings = GenerateEmbedings(summarized_text, image_summary, tables_summary)
#         retriver = embeddings.embed()

#         agent = QA(retriver)
        
#         return agent

from app.utils.text_extractor import ExtractionAndChunking
from app.utils.summarizer import GeminiSummarizer
from app.utils import file_handler
from app.utils.Cohere_Embedding import GenerateEmbedings, context_generation
from app.utils.images_tables_extract import ImageTable
from app.utils.langchain_handler import QA
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

summarizer_handler = GeminiSummarizer()

class Pipeline:
    def __init__(self):
        pass

    def common(self, file):
        self.uploaded_file = file
        try:
            file_path = file_handler.save(self.uploaded_file)
            logger.info(f"File saved successfully: {file_path}")

            extractor = ExtractionAndChunking(file_path)
            extracted_text = extractor.extract_text_from_pdf_pymupdf()
            
            structured_text = extractor.model(extracted_text)

            logger.info("Text extraction and structuring completed.")
            return structured_text
        except Exception as e:
            logger.error(f"Error during text extraction: {e}")
            raise RuntimeError("Failed to extract text from the uploaded file.")

    def answer_generate(self, structured_text, uploaded_file):
        try:
            self.uploaded_file = uploaded_file
            summarized_text = summarizer_handler.summarize(structured_text)

            image_table = ImageTable(self.uploaded_file)
            images, tables = image_table.extract_images_and_tables_from_pdf()

            context = context_generation(summarized_text)

            image_summary = summarizer_handler.Image_summarize(context, images)
            tables_summary = summarizer_handler.Table_summarize(context, tables)

            embeddings = GenerateEmbedings(summarized_text, image_summary, tables_summary)
            retriever = embeddings.embed()

            agent = QA(retriever)

            logger.info("QA pipeline setup completed.")
            return agent
        except Exception as e:
            logger.error(f"Error during QA pipeline setup: {e}")
            raise RuntimeError("Failed to set up the QA pipeline.")