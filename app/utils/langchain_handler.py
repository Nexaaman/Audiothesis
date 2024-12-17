#from langchain.prompts import PromptTemplate
#from langchain_huggingface import HuggingFaceEndpoint  

# class LangChainManager:
#     def __init__(self):
#         self.api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#         self.llm_chain = self.create_llm_chain("facebook/bart-large-cnn")  

#     def create_llm_chain(self, model: str):
#         prompt_template = PromptTemplate.from_template(
#             "Summarize the following text while maintaining maximum context: \n{text}\n\nSummary:"
#         )
#         print(model)
#         llm = HuggingFaceEndpoint(
#             repo_id=model,
#             huggingfacehub_api_token=self.api_token,
#             max_new_tokens=2048,
#             temperature=0.5,
#         )
        
#         chain = prompt_template | llm
#         return chain

#     def summarize(self, text: str, model: str = "bart"):
#         print(self.api_token)
#         if model.lower() == "t5":
#             self.llm_chain = self.create_llm_chain("t5-base")
#         else:
#             self.llm_chain = self.create_llm_chain("facebook/bart-large-cnn")
        
#         summary = self.llm_chain.invoke({"text": text})
        
#         return summary


