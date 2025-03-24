# multimodal_rag_query_handler.py
import os
import mimetypes
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, OpenCLIPEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEndpoint
from PyPDF2 import PdfReader

# Paths
TEXT_DB_PATH = "vectorstore/db_faiss"
IMAGE_DB_PATH = "vectorstore/image_faiss"

# Load Embedding Models
text_embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
image_embedder = OpenCLIPEmbeddings(model_name="openclip-ViT-B-32")

# Load Vector Stores
text_db = FAISS.load_local(TEXT_DB_PATH, text_embedder, allow_dangerous_deserialization=True)
image_db = FAISS.load_local(IMAGE_DB_PATH, image_embedder, allow_dangerous_deserialization=True)

# Load LLM
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=HUGGINGFACE_REPO_ID, model_kwargs={"token": HF_TOKEN, "max_length": "512"})

# Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, say you don't know. Don't make things up.

Context: {context}
Question: {question}
"""
prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# Retrieval Chain
text_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=text_db.as_retriever(), chain_type_kwargs={"prompt": prompt})
image_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=image_db.as_retriever(), chain_type_kwargs={"prompt": prompt})

# === Query Router ===
def handle_input(user_input=None, file_path=None):
    if user_input:
        print("🔍 Text Query Detected")
        response = text_chain.invoke({"query": user_input})
        return response

    elif file_path:
        mime, _ = mimetypes.guess_type(file_path)

        if mime and mime.startswith("image"):
            print("🖼 Image Input Detected")
            dummy_question = "What can be inferred from this medical image?"
            image_doc = Document(page_content=dummy_question, metadata={"image_path": file_path})
            response = image_chain.invoke({"query": dummy_question})
            return response

        elif mime and mime == "application/pdf":
            print("📄 PDF Detected — extracting text...")
            reader = PdfReader(file_path)
            pdf_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            if not pdf_text.strip():
                return {"result": "No text found in PDF."}
            dummy_question = "Summarize the document."
            response = text_chain.invoke({"query": dummy_question, "context": pdf_text})
            return response

        else:
            return {"result": "Unsupported file format."}

    else:
        return {"result": "No valid input provided."}

# === Example Run ===
if __name__ == "__main__":
    # Option 1: Text
    print(handle_input(user_input="What are common signs of pneumonia in chest X-rays?"))

    # Option 2: Image File
    # print(handle_input(file_path="data/images/medpix_0001.jpg"))

    # Option 3: PDF File
    # print(handle_input(file_path="data/pdfs/sample_medical_paper.pdf"))
