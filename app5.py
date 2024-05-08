from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

# Set page title and layout
st.set_page_config(
    page_title="BuddyBOT",
    page_icon=":robot_face:",
    layout="wide"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .message {
        padding: 10px;
        border-radius: 10px;
        margin: 10px;
    }
    .user-message {
        background-color: #E2E2E2;
    }
    .bot-message {
        background-color: #3d89eb;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define title HTML for the app
title_html = """
   <style>
       .title {
           color: #3d89eb;
           text-align: center;
       }
   </style>
   <h1 class="title">🤖 BuddyBOT</h1>
"""

# Display title HTML
st.markdown(title_html, unsafe_allow_html=True)

# Initialize chat session history
if "chat_session" not in st.session_state:
    st.session_state.chat_session = []

# Function to load documents locally
def load_docs_locally(files):
    data = []
    # Loading documents based on their extension
    for file in files:
        # Check the file extension and use the appropriate loader
        _, extension = os.path.splitext(file)
        if extension == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file)
        elif extension == ".txt":
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(file, encoding="utf-8")
        elif extension == ".docx":
            from langchain_community.document_loaders import Docx2textLoader
            loader = Docx2textLoader(file)
        else:
            print(f"No loader available for file format: {extension}")
        data += loader.load()
    return data

# Function to chunk data
def chunk_data(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    text = "\n".join([doc.page_content for doc in docs])
    chunks = text_splitter.split_text(text)
    return chunks

# Function to embed data using FAISS
def embed_data(chunks):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector_index = FAISS.from_texts(chunks, embedding).as_retriever(search_type="similarity")
    except Exception as e:
        st.error(f"An error occurred during data embedding: {str(e)}")
        vector_index = None
    return vector_index

# Remaining code remains the same