import pymupdf
import os
from groq import Groq, APIError
import string
import time
import json
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field
load_dotenv()

class Section(BaseModel):
    section_name: str = Field(description="Name of the section")
    section_text: str = Field(description="Text content of the section")


parser = JsonOutputParser(pydantic_object=Section)

class ExtractionAndChunking:
    def __init__(self, file_path: str):
        self.file = file_path
        self.client = Groq()

    def extract_text_from_pdf_pymupdf(self):

        extracted_text = ""

        with pymupdf.open(self.file) as pdf:
            for page_num in range(pdf.page_count):
                page = pdf.load_page(page_num)
                extracted_text += page.get_text("text") + "\n"

        return extracted_text


    def split_text_into_chunks(self,text, max_token_size=3500, overlap=500):
        self.text = text
        words = self.text.split()
        chunks = []
        current_chunk = []

        token_count = 0
        for word in words:
            token_count += 1
            current_chunk.append(word)

            if token_count >= max_token_size:
                chunks.append(" ".join(current_chunk))
                # Add overlap for the next chunk
                current_chunk = current_chunk[-overlap:]
                token_count = len(current_chunk)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def process_response(self, parsed_response):
      """Processes the parsed response safely."""
      try:
          print("Parsed Response:", parsed_response)

          sections = {}
          if isinstance(parsed_response, list):
              for entry in parsed_response:
                  section_name = entry.get('section_name', '').strip()
                  section_text = entry.get('section_text', '').strip()
                  if section_name and section_text:
                      sections[section_name] = section_text
          elif isinstance(parsed_response, dict):
              section_name = parsed_response.get('section_name', '').strip()
              section_text = parsed_response.get('section_text', '').strip()
              if section_name and section_text:
                  sections[section_name] = section_text

          return sections
      except Exception as e:
          print(f"Unexpected error in process_response: {e}")
          return {}



    def model(self, extracted_text:str):

        self.extracted_text = extracted_text
        self.text_chunks = self.split_text_into_chunks(self.extracted_text)

        self.all_sections = {}

        api_key = os.environ.get("GROQ_API_KEY")
        token_usage = 0

        client = Groq(api_key=api_key)

        for idx, chunk in enumerate(self.text_chunks):
            print(f"Processing chunk {idx + 1}/{len(self.text_chunks)}")

            system_prompt = """You are an expert language model trained on a variety of texts. Your task is to analyze the extracted text from a research paper and identify its key sections.
            Identify sections such as 'Abstract', 'Introduction', 'Methods', 'Results', 'Conclusion', or any other sections based on the content. It is critically important that your response strictly adheres to the JSON format provided."""

            user_prompt = f"""Here is the extracted text from the research paper:
            {chunk}

            Please return the section names and the corresponding text for each section in valid JSON format as follows:
            [
                {{
                    "section_name": "Abstract",
                    "section_text": "The dominant sequence transduction models are based on complex neural networks..."
                }},
                {{
                    "section_name": "Introduction",
                    "section_text": "In recent years, there has been significant progress in..."
                }}
            ]
            """

            try:
                response = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[
                        {'role': "system", "content": system_prompt},
                        {'role': "user", "content": user_prompt}
                    ],
                    temperature=0
                )
                token_count = len(system_prompt.split()) + len(user_prompt.split()) + len(response.choices[0].message.content.split())
                token_usage += token_count
                print(f"Tokens used in this request: {token_count}, Total so far: {token_usage}")

                parsed_sections = parser.parse(response.choices[0].message.content)
                print(parsed_sections)
                chunk_sections = self.process_response(parsed_sections)


                for section, content in chunk_sections.items():
                    if section in self.all_sections:
                        self.all_sections[section] += "\n" + content
                    else:
                        self.all_sections[section] = content

            except APIError as e:
                if 'rate_limit_exceeded' in str(e):
                    print("Rate limit exceeded, waiting 60 seconds and retrying...")
                    time.sleep(60)
                    continue
                raise
            except Exception as e:
                print(f"Failed to process chunk {idx + 1}: {e}")

            time.sleep(10)

        return self.all_sections