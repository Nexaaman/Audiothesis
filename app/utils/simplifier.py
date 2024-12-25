import os
import time
import json
from typing import Dict, List
from groq import Groq
import re
class ScriptGenerator:
    def __init__(self):
        pass

    def process_response(self, response_content: str) -> Dict[str, str]:
        try:
            cleaned_response = re.sub(r"```[\w]*", "", response_content).strip()
            parsed_response = json.loads(cleaned_response)
            
            sections = {}
            if isinstance(parsed_response, list):
                for entry in parsed_response:
                    section_name = entry.get('episode title', '').strip()
                    section_text = entry.get('script content', '').strip()
                    if section_name and section_text:
                        sections[section_name] = section_text
            elif isinstance(parsed_response, dict):
                section_name = parsed_response.get('episode title', '').strip()
                section_text = parsed_response.get('script content', '').strip()
                if section_name and section_text:
                    sections[section_name] = section_text

            return sections

        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print("Raw response content:", response_content)
            return {}

    def split_text_into_chunks(self, text: str, max_token_size: int = 3500, overlap: int = 500) -> List[str]:
        """Splits text into chunks for processing."""
        words = text.split()
        chunks = []
        current_chunk = []

        token_count = 0
        for word in words:
            token_count += 1
            current_chunk.append(word)

            if token_count >= max_token_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap:]
                token_count = len(current_chunk)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


    def generate_podcast_episodes(self, sections: Dict[str, str]) -> Dict[str, str]:
        """Generates detailed podcast episodes based on input research content."""
        api_key = os.environ.get("GROQ_API_KEY")
        client = Groq(api_key=api_key)

        episodes = {}
        token_usage = 0

       
        content_string = "\n".join([f"{section}: {text}" for section, text in sections.items()])
        chunks = self.split_text_into_chunks(content_string, max_token_size=3500, overlap=500)

        system_prompt = """
        You are a professional podcast scriptwriter. I need you to create a 5-minute podcast episode based on the following research content.

        ### Requirements:
        - Tone: Conversational, simple, and engaging (targeting a general audience).
        - Decide the number of episodes based on content length (5 episodes for shorter content, 7 for longer content).
        - Content to cover:
        1. An engaging hook to grab attention.
        2. A clear breakdown of the section in layman's terms.
        3. Use relatable examples if necessary.
        4. Conclude with a teaser for the next episode.
        """

        for idx, chunk in enumerate(chunks):
            print(f"Processing chunk {idx + 1}/{len(chunks)}")
            user_prompt = f"""
            Below is a part of the research content:
            {chunk}

            Based on this content, create detailed podcast episodes. For each episode, include:
            - A descriptive title.
            - A script with a clear introduction, in-depth explanation, and engaging conclusion.

            
            Please return the episode title and the script content for each episode in valid JSON format like this:
            [
                {{
                    "episode title": "Episode 1: The Big Picture - Understanding the Abstract",
                    "script content": "Welcome to our first episode! Today, we explore the abstract of this groundbreaking research..."
                }},
                {{
                    "episode title": "Episode 2: Laying the Foundation - The Introduction",
                    "script content": "In this episode, we dive into the introduction, covering the background and the research motivation..."
                }}
            ]
            """

            try:
                
                response = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.8,  
                    max_tokens=4000
                )

                
                response_text = response.choices[0].message.content.strip()
                processed_response = self.process_response(response_text)

                token_count = len(system_prompt.split()) + len(user_prompt.split()) + len(response_text.split())
                token_usage += token_count
                print(f"Tokens used in this request: {token_count}, Total so far: {token_usage}")

                
                for title, content in processed_response.items():
                    if title in episodes:
                        episodes[title] += "\n" + content
                    else:
                        episodes[title] = content

            except Exception as e:
                print(f"Failed to process chunk {idx + 1}: {e}")

           
            time.sleep(5)

        return episodes
