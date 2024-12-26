from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from base64 import b64decode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

class QA:

    def __init__(self, retriever):
        self.retriever = retriever

    def parse_docs(self, docs):

        b64 = []
        text = []
        for doc in docs:
            try:
                b64decode(doc)
                b64.append(doc)
            except Exception as e:
                text.append(doc)
        return {"images": b64, "texts": text}

    def build_prompt(self, kwargs):
      docs_by_type = kwargs["context"]
      user_question = kwargs["question"]

      context_text = ""

      if len(docs_by_type["texts"]) > 0:
          for text_element in docs_by_type["texts"]:
              context_text += text_element  # Fixed: text_element is already a string

      prompt_template = f"""
      Answer the question based only on the following context, which can include text, tables, and the below image.
      Context: {context_text}
      Question: {user_question}
      """
      prompt_content = [{"type": "text", "text": prompt_template}]

      if len(docs_by_type["images"]) > 0:
          for image in docs_by_type["images"]:
              prompt_content.append(
                  {
                      "type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                  }
              )

      return ChatPromptTemplate.from_messages(
          [
              HumanMessage(content=prompt_template),  # Pass the prompt_template string directly
          ]
      )

    def response(self, Question: str, option: bool):

        self.Question = Question
        self.option = option
        chain = (
            {
                "context": self.retriever | RunnableLambda(self.parse_docs),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self.build_prompt)
            | ChatGroq(model="llama-3.1-8b-instant", api_key = os.environ.get("GROQ_API_KEY"))
            | StrOutputParser()
        )

        chain_with_sources = {
            "context": self.retriever| RunnableLambda(self.parse_docs),
            "question": RunnablePassthrough(),
        } | RunnablePassthrough().assign(
            response=(
                RunnableLambda(self.build_prompt)
                | ChatGroq(model="llama-3.1-8b-instant",  api_key = os.environ.get("GROQ_API_KEY"))
                | StrOutputParser()
            )
        )

        if self.option == True:
            response = chain_with_sources.invoke(Question)
        else:
            response = chain.invoke(Question)

        return response
