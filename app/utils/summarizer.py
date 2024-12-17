import os
from dotenv import load_dotenv
import time
from typing import List, Dict
import google.generativeai as genai
load_dotenv()

class GeminiSummarizer:
    def __init__(self):
        self.api_token = os.getenv("GEMINI_API_KEY")
        self.REQUESTS_PER_MINUTE = 2
        self.TOKENS_PER_MINUTE = 32000
        self.REQUESTS_PER_DAY = 50

        self.request_count = 0
        self.token_usage = 0

         
    
    
    def summarize_all_sections_in_one_call(self,sections: Dict[str, str]) -> Dict[str, str]:
        
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
            return self.summarize_all_sections_in_one_call(sections)
        except Exception as e:
            print(f"Failed to summarize all sections: {e}")
            return {section_name: "Error in summarization." for section_name in sections}


    def save_summarized_output(self, summarized_sections: Dict[str, str]):
        
        with open("outputs/summarized_output.txt", "w") as f:
            for section, summary in summarized_sections.items():
                f.write(f"**{section}**\n")
                f.write(f"{summary}\n\n")
        print("Summarized output written successfully to summarized_output.txt")
