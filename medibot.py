import os
import streamlit as st
from huggingface_hub import InferenceClient

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

class CustomInferenceLLM:
    def __init__(self, model_id, hf_token, temperature=0.5, max_tokens=512):
        self.client = InferenceClient(model=model_id, token=hf_token)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _call(self, prompt):
        response = self.client.text_generation(
            prompt=prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature
        )
        return response

    def invoke(self, input):
        context = input.get("context", "")
        question = input.get("question", "")
        full_prompt = f"Use the pieces of information provided in the context to answer user's question.\n\nContext: {context}\nQuestion: {question}\nStart the answer directly. No small talk please."
        return {"result": self._call(full_prompt), "source_documents": []}

def format_source_documents(docs):
    formatted_text = ""
    for doc in docs:
        content = doc.page_content.strip()
        metadata = doc.metadata
        source_info = f"📄 Source: {metadata.get('source', 'Unknown')}, Page: {metadata.get('page', 'N/A')}"
        formatted_text += f"\n{source_info}\n{content}\n{'-'*80}\n"
    return formatted_text

def main():
    st.title("📚 Ask Your PDF Chatbot!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Type your question here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        HUGGINGFACE_REPO_ID = "tiiuae/falcon-7b-instruct"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load the vector store.")
                return

            llm = CustomInferenceLLM(model_id=HUGGINGFACE_REPO_ID, hf_token=HF_TOKEN)
            retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
            context_docs = retriever.get_relevant_documents(prompt)
            context_text = "\n".join([doc.page_content.strip() for doc in context_docs])

            response = llm.invoke({"context": context_text, "question": prompt})
            result = response["result"]

            formatted_sources = format_source_documents(context_docs)

            final_answer = f"💬 **Answer:**\n{result.strip()}\n\n📚 **Relevant Source Content:**\n{formatted_sources}"

            st.chat_message('assistant').markdown(final_answer)
            st.session_state.messages.append({'role': 'assistant', 'content': final_answer})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
