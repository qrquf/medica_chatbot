from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import time
import os
import sys

# Step 1: Load raw PDF(s)
DATA_PATH = "data/"

def load_pdf_files(data):
    print("\n🔍 Loading PDF files...")
    all_documents = []
    pdf_files = [f for f in os.listdir(data) if f.endswith(".pdf")]
    for pdf in tqdm(pdf_files, desc="Loading PDFs"):
        print(f"📄 Loading {pdf} ...")
        loader = PyMuPDFLoader(os.path.join(data, pdf))
        documents = loader.load()
        all_documents.extend(documents)
    print(f"✅ Loaded {len(all_documents)} pages from {len(pdf_files)} PDFs.")
    return all_documents

# Step 2: Create Chunks
def create_chunks(extracted_data):
    print("\n✂ Creating text chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    print(f"✅ Created {len(text_chunks)} text chunks.")
    return text_chunks

# Step 3: Create Vector Embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

# Step 4: Store embeddings in FAISS with progress bar
def store_faiss(text_chunks, embedding_model):
    print("\n🧠 Embedding chunks and saving to FAISS DB...")
    progress_bar = tqdm(text_chunks, desc="Embedding Chunks")
    db = FAISS.from_documents(list(progress_bar), embedding_model)
    progress_bar.close()
    DB_FAISS_PATH = "vectorstore/db_faiss"
    db.save_local(DB_FAISS_PATH)
    print("✅ FAISS vectorstore saved successfully at:", DB_FAISS_PATH)

# Main execution
if __name__ == "__main__":
    start_time = time.time()
    documents = load_pdf_files(DATA_PATH)
    text_chunks = create_chunks(documents)
    embedding_model = get_embedding_model()
    store_faiss(text_chunks, embedding_model)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n✅ Done in {total_time:.2f} seconds (~{total_time/60:.2f} minutes).")

    # Force clean exit
    sys.exit()
