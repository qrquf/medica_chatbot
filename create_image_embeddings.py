# create_image_embeddings.py
import os
import json
from PIL import Image
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenCLIPEmbeddings

# CONFIG
IMAGE_FOLDER = "data/images/"
META_FILE = "data/image_meta.json"
DB_FAISS_PATH = "vectorstore/image_faiss"

# Load Metadata
with open(META_FILE, "r") as f:
    image_metadata = json.load(f)

# Load Image Embedding Model (CLIP-based)
embedding_model = OpenCLIPEmbeddings(model_name="openclip-ViT-B-32")

# Prepare LangChain documents for FAISS
documents = []
from langchain.schema import Document

for meta in tqdm(image_metadata, desc="🔗 Generating Image Embeddings"):
    image_path = os.path.join(IMAGE_FOLDER, meta["image_file"])
    try:
        # Generate caption text as document content
        text_content = f"Image Caption: {meta['caption']}\nDiagnosis: {meta['diagnosis']}"

        doc = Document(
            page_content=text_content,
            metadata={"image_path": image_path, "source_url": meta["source_url"]}
        )
        documents.append(doc)

    except Exception as e:
        print(f"Skipping image {meta['image_file']}: {e}")

# Embed and store in FAISS
print("\n📦 Embedding all image descriptions and saving to FAISS DB...")
db = FAISS.from_documents(documents, embedding_model)
db.save_local(DB_FAISS_PATH)
print("✅ Image vectorstore saved at:", DB_FAISS_PATH)
