# multimodal_chatbot_app.py
import os
import streamlit as st
from multimodal_rag_query_handler import handle_input
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="🧠 Multimodal Medical Chatbot", layout="wide")
st.title("🧠 Multimodal Medical Chatbot (Text, PDF, Image)")

st.markdown("""
Upload a **PDF**, **Medical Image (X-ray, CT, Ultrasound, Skin Lesion, etc.)**, or type your medical question below. The system will automatically process and respond using a RAG-enabled LLM.
""")

# File upload section
uploaded_file = st.file_uploader("📎 Upload PDF or Medical Image", type=["pdf", "jpg", "jpeg", "png"])

# Text input section
user_query = st.text_input("💬 Or type your question below:", "")

submit = st.button("🔍 Submit Query")

if submit:
    if user_query:
        st.info("Processing your text query...")
        result = handle_input(user_input=user_query)
        st.markdown(f"**💬 Answer:**\n\n{result['result']}")

    elif uploaded_file:
        file_path = os.path.join("temp_uploads", uploaded_file.name)
        os.makedirs("temp_uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        st.info(f"Processing your uploaded file: {uploaded_file.name}")
        result = handle_input(file_path=file_path)
        st.markdown(f"**💬 Answer:**\n\n{result['result']}")

    else:
        st.warning("Please upload a file or type a question to begin.")