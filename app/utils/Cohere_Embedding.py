from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, List
from dotenv import load_dotenv
import os
import uuid
from langchain_pinecone import PineconeVectorStore
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain.retrievers import MultiVectorRetriever
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone,ServerlessSpec
load_dotenv()

def context_generation(summary: Dict[str, str]) -> str:

    summary_str = str(summary).replace("{", "{{").replace("}", "}}")


    prompt_text = f"""This is the extracted summary of a research paper: {summary_str}.
                Based on this summary, generate a brief context in 4 to 5 lines,
                highlighting the key focus and significance of the research."""

    model = ChatCohere(
        temperature=0,
        model="command-r-08-2024",
        cohere_api_key=os.environ.get("COHERE_API_KEY")
    )
    prompt = ChatPromptTemplate.from_template(prompt_text)

    context_chain = prompt | model | StrOutputParser()
    context = context_chain.invoke({"summary": summary_str})

    return context

class GenerateEmbedings():
    def __init__(self,text_summary: Dict[str, str], image_summaries: List[str] , table_summaries: List[str]):
        self.text_summary = text_summary
        self.image_summaries = image_summaries
        self.table_summaries = table_summaries

    def embed(self):
        
        index_name = "multi-modal-rag2"

        embeddings = CohereEmbeddings(model="embed-english-light-v3.0")  
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )

        store = InMemoryStore()
        id_key = "doc_id"

        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
        )

        #For Text
        doc_ids = [str(uuid.uuid4()) for _ in self.text_summary]
        summary_texts = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(self.text_summary)
        ]
        retriever.vectorstore.add_documents(summary_texts)
        retriever.docstore.mset(list(zip(doc_ids, self.text_summary)))

        #For Tables

        table_ids = [str(uuid.uuid4()) for _ in self.table_summaries]
        summary_tables = [
            Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(self.table_summaries)
        ]
        retriever.vectorstore.add_documents(summary_tables)
        retriever.docstore.mset(list(zip(table_ids, self.table_summaries)))

        #For Images

        img_ids = [str(uuid.uuid4()) for _ in self.image_summaries]
        summary_img = [
            Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(self.image_summaries)
        ]
        retriever.vectorstore.add_documents(summary_img)
        retriever.docstore.mset(list(zip(img_ids, self.image_summaries)))

        return retriever