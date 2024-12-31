from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from typing import Dict, List
from langchain_core.messages import SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate
import os
from langchain_core.output_parsers import JsonOutputParser

class ScriptGenerator:
    def __init__(self):
        pass

    def process_response(self, parsed_response):
        try:
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
        except Exception as e:
            print(f"Unexpected error in process_response: {e}")
            return {}

    def generate_podcast_episodes(self, content: Dict[str, str]) -> Dict[str, str]:

        self.episodes = {}

        chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        """
                        You are a professional podcast scriptwriter. I need you to create engaging and detailed podcast episodes based on the following research content.

                        ### Requirements:
                        - Tone: Conversational, simple, and engaging (targeting a general audience).
                        - Decide the number of episodes based on content length (5–7 episodes depending on content length).
                        - Target duration: Each script should provide enough content for at least 2  minutes of audio (~2000 characters).
                        - Content to cover:
                        1. An engaging hook to grab attention.
                        2. A deep dive into the topic with clear explanations in layman's terms.
                        3. Use relatable examples, anecdotes, or analogies to make the topic engaging.
                        4. Summarize key takeaways.
                        5. Conclude with a teaser for the next episode.
                        """
                    )
                ),
                HumanMessagePromptTemplate.from_template(
                    """
                    Below is a part of the research content:
                    {content}

                    Based on this content, create detailed podcast episodes. For each episode:
                    - Include an engaging title.
                    - Provide a comprehensive script with a clear introduction, in-depth explanation, and engaging conclusion.
                    - Ensure the script is at max 2000 characters.

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
                    ]"""
                ),
            ]
        )
        model = ChatCohere(
            temperature=0.7,
            model="command-r-plus-08-2024",
            cohere_api_key=os.environ.get("COHERE_API_KEY")
        )
        script_chain = chat_template | model | JsonOutputParser()
        script = script_chain.invoke({"content": content})
        episodes = self.process_response(script)
        return episodes