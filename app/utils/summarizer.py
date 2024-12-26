import os
from dotenv import load_dotenv
import time
from langchain_cohere import ChatCohere
from typing import List, Dict
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import google.api_core.exceptions
load_dotenv()

class GeminiSummarizer:
    def __init__(self):
        self.api_token = os.getenv("GEMINI_API_KEY")
        self.REQUESTS_PER_MINUTE = 2
        self.TOKENS_PER_MINUTE = 32000
        self.REQUESTS_PER_DAY = 50

        self.request_count = 0
        self.token_usage = 0

    def Image_summarize(self, context: str, images: list) -> list:
        self.context = context
        self.images = images
        self.image_summaries = []

        prompt_template = """Describe the image in detail. For context,
                  the image is part of a research paper with object: {context}. Be specific about graphs, such as bar plots."""

        messages = [
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image}"},
                    },
                ],
            )
        ]
        prompt = ChatPromptTemplate.from_messages(messages)

        model = ChatGoogleGenerativeAI(temperature=0.5, model="gemini-1.5-pro", api_key=self.api_token)
        summarize_chain =  prompt | model | StrOutputParser()

        for image in images:
            try:
                image_summary = summarize_chain.invoke({"context": context,  "image": image})
                self.image_summaries.append(image_summary)

                time.sleep(2)
            except google.api_core.exceptions.ResourceExhausted:
                print(f"Rate limit exceeded. Waiting for 60 seconds before retrying...")
                time.sleep(60)
                image_summary = summarize_chain.invoke({"context": context,  "image": image})
                self.image_summaries.append(image_summary)

        return self.image_summaries

    def Table_summarize(self, context: str, tables: list) -> list:
      self.context = context
      self.tables = tables
      self.table_summaries = []

      # Define the prompt template for table summarization
      prompt_template = """Describe the table in detail. For context,
                    the table is part of a research paper with objective: {context}.
                    Be specific about the contents and highlight key insights.

                    Here's the table data:
                    {table}"""

      
      prompt = ChatPromptTemplate.from_template(prompt_template)

      # Initialize the LLM model
      model = ChatCohere(temperature=0.5,
        model="command-r-08-2024",
        cohere_api_key=os.environ.get("COHERE_API_KEY"))
      summarize_chain = prompt | model | StrOutputParser()

      # Process each table in the list
      for table in tables:
          try:
              # Convert table content to string and format for LLM
              table_content = str(table['content']).replace("{", "{{").replace("}", "}}")
              table_content = str(table['content']).replace("{", "{{").replace("}", "}}")
              # Generate summary for the table
              table_summary = summarize_chain.invoke({"context": context, "table": table_content})
              self.table_summaries.append(table_summary)

              # Avoid rapid successive API calls
              time.sleep(2)
          except google.api_core.exceptions.ResourceExhausted:
              print(f"Rate limit exceeded. Waiting for 60 seconds before retrying...")
              time.sleep(60)
              table_summary = summarize_chain.invoke({"context": context, "table": table_content})
              self.table_summaries.append(table_summary)

      return self.table_summaries

      
    
    
    def text_summarize(self,sections: Dict[str, str]) -> Dict[str, str]:

        #global request_count, token_usage

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        system_prompt = (
            "You are a highly intelligent assistant trained to summarize research papers. "
            "Your goal is to retain critical details, context, and format while providing concise and accurate summaries."
        )

        combined_section_prompts = []
        for section_name, section_text in sections.items():
            combined_section_prompts.append(
                f"Section: {section_name}\n\n{section_text}\n\n"
                "Please provide a concise summary for this section."
            )
        user_prompt = "\n---\n".join(combined_section_prompts)
        combined_prompt = system_prompt + "\n\n" + user_prompt

        input_token_count = len(combined_prompt.split())

        if self.request_count >= self.REQUESTS_PER_DAY:
            print("Daily request limit reached. Exiting.")
            return {section_name: "Error: Daily request limit reached." for section_name in sections}

        if self.token_usage + input_token_count > self.TOKENS_PER_MINUTE:
            print("Token usage limit reached for this minute. Waiting...")
            time.sleep(60)

        if self.request_count >= self.REQUESTS_PER_MINUTE:
            print("Request limit reached for this minute. Waiting...")
            time.sleep(60)

        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(
                combined_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=4000,
                    temperature=0.7,
                )
            )

            summaries = {}
            response_text = response.text.strip()
            generated_summaries = response_text.split("\n---\n")

            if len(generated_summaries) != len(sections):
                print("Warning: Number of summaries does not match the number of sections. Verify output.")

            for (section_name, _), summary in zip(sections.items(), generated_summaries):
                summaries[section_name] = summary.strip()

            self.request_count += 1
            self.token_usage += input_token_count + len(response.text.split())

        except Exception as e:
            print(f"API call failed for all sections: {e}")
            return {section_name: "Error in summarization." for section_name in sections}

        return summaries

    def summarize(self,sections: Dict[str, str]) -> Dict[str, str]:
        print(f"Summarizing all sections at once.")
        try:
            return self.text_summarize(sections)
        except Exception as e:
            print(f"Failed to summarize all sections: {e}")
            return {section_name: "Error in summarization." for section_name in sections}


    def save_summarized_output(self, summarized_sections: Dict[str, str]):

        with open("outputs/summarized_output.txt", "w") as f:
            for section, summary in summarized_sections.items():
                f.write(f"**{section}**\n")
                f.write(f"{summary}\n\n")
        print("Summarized output written successfully to summarized_output.txt")
