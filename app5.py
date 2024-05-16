from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS 
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
import os
from dotenv import load_dotenv ,find_dotenv
load_dotenv(find_dotenv(), override=True)
if os.environ:
    for api_key in os.environ:
        if "API_KEY" in api_key:
            print(api_key)
else:
    import getpass
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("GOOGLE_API_KEY")

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
    for file in files:
        if not file.startswith("."):
            _, extension = os.path.splitext(file)
            if extension == ".pdf":
                from langchain.document_loaders import PyPDFLoader 
                loader = PyPDFLoader(file)
            elif extension == ".txt":
                from langchain.document_loaders import TextLoader 
                loader = TextLoader(file, encoding="utf-8")
            elif extension == ".docx":
                from langchain.document_loaders import Docx2textLoader
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
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.vectorstores import FAISS  # Modification de l'import
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from dotenv import load_dotenv ,find_dotenv
load_dotenv(find_dotenv(), override=True)
if os.environ:
    for api_key in os.environ:
        if "API_KEY" in api_key:
            print(api_key)
else:
    import getpass
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("GOOGLE_API_KEY")

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
        if not file.startswith("."):
            _, extension = os.path.splitext(file)
            if extension == ".pdf":
                from langchain.document_loaders import PyPDFLoader 
                loader = PyPDFLoader(file)
            elif extension == ".txt":
                from langchain.document_loaders import TextLoader 
                loader = TextLoader(file, encoding="utf-8")
            elif extension == ".docx":
                from langchain.document_loaders import Docx2textLoader
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
    # Get Google API Key from environment variable
    google_api_key = os.getenv("GOOGLE_API_KEY")

    # Check if Google API Key is provided
    if google_api_key is None:
        raise ValueError("Google API Key is missing. Please set the GOOGLE_API_KEY environment variable.")
    #google_api_key ="AIzaSyD7QVZAzF7HtoOgDJnlkHGvsJ2l6FChYrY"
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)  # Pass the Google API Key as a named parameter
    vector_index = FAISS.from_texts(chunks, embedding).as_retriever(search_type="similarity")  # Modification de l'utilisation de FAISS
    return vector_index

# Remaining code remains the same

# Function to ask a question to the chatbot
def ask_question(query, vector_index):
    template = """
    use the following pieces of context to answer the question at the end, if you don't know the answer, simply respond that you don't know. Keep the answer concise.
    {context}
    Question: {question}
    """
    QA_CHAIN_TEMPLATE = PromptTemplate.from_template(template)
    faiss_chain = RetrievalQA.from_chain_type(llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=1), retriever=vector_index, return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_TEMPLATE})

    response = faiss_chain({"query": query})
    result = response["result"]
    return result

# Function to clear chat history
def clear_history():
    st.session_state.chat_session = []

# Function to start a new chat session
def start_new_chat():
    st.session_state.chat_session = []
    st.success("New chat session started. You can now chat with the Chatbot.")

# Function to toggle between light and dark themes
def toggle_theme():
    theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.write(
            """
            <style>
            .stApp { background-color: #333333;}
            .stSidebar { background-color: #333333;}
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.write(
            """
            <style>
            .stApp { background-color: #FFFFFF;}
            .stSidebar { background-color: #FFFFFF;}
            </style>
            """,
            unsafe_allow_html=True,
        )

# List of files to load
files = ["files/Banque_FR.pdf", "files/profe.txt"]
docs = load_docs_locally(files)
chunks = chunk_data(docs)
vector_index = embed_data(chunks)

# Sidebar to toggle between light and dark modes and start a new chat session
st.sidebar.title("Display Settings")
toggle_theme()
if st.sidebar.button("Start New Chat"):
    start_new_chat()

# Main chat interface
# st.write("Chat with the Chatbot:")
user_input = st.text_input("Chat with the Chatbot:", key="user_input")
if st.button("Ask"):
    response = ask_question(user_input, vector_index)
    st.markdown(f'<div class="message user-message">{user_input}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="message bot-message">Bot: {response}</div>', unsafe_allow_html=True)

# Display chat history in the main section
st.write("Chat History:")
for chat in st.session_state.chat_session:
    if chat["role"] == "user":
        st.markdown(f'<div class="message user-message">You: {chat["context"]}</div>', unsafe_allow_html=True)
    elif chat["role"] == "assistant":
        st.markdown(f'<div class="message bot-message">Bot: {chat["context"]}</div>', unsafe_allow_html=True)

# Button to clear chat history
if st.button("Clear History"):
    clear_history()
    st.success("Chat history has been cleared.")

# Function to ask a question to the chatbot
def ask_question(query, vector_index):
    template = """
    use the following pieces of context to answer the question at the end, if you don't know the answer, simply respond that you don't know. Keep the answer concise.
    {context}
    Question: {question}
    """
    QA_CHAIN_TEMPLATE = PromptTemplate.from_template(template)
    faiss_chain = RetrievalQA.from_chain_type(llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=1), retriever=vector_index, return_source_documents=True, chain_type_kwargs={"prompt": QA_CHAIN_TEMPLATE})

    response = faiss_chain({"query": query})
    result = response["result"]
    return result

# Function to clear chat history
def clear_history():
    st.session_state.chat_session = []

# Function to start a new chat session
def start_new_chat():
    st.session_state.chat_session = []
    st.success("New chat session started. You can now chat with the Chatbot.")

# Function to toggle between light and dark themes
def toggle_theme():
    theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.write(
            """
            <style>
            .stApp { background-color: #333333;}
            .stSidebar { background-color: #333333;}
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.write(
            """
            <style>
            .stApp { background-color: #FFFFFF;}
            .stSidebar { background-color: #FFFFFF;}
            </style>
            """,
            unsafe_allow_html=True,
        )

# List of files to load
files = ["files/Banque_FR.pdf","files/banque_AR.pdf","files/profe.txt"]
docs = load_docs_locally(files)
chunks = chunk_data(docs)
vector_index = embed_data(chunks)

# Sidebar to toggle between light and dark modes and start a new chat session
st.sidebar.title("Display Settings")
toggle_theme()
if st.sidebar.button("Start New Chat"):
    start_new_chat()

# Main chat interface
# st.write("Chat with the Chatbot:")
user_input = st.text_input("Chat with the Chatbot:", key="user_input")
if st.button("Ask"):
    response = ask_question(user_input, vector_index)
    st.markdown(f'<div class="message user-message">{user_input}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="message bot-message">Bot: {response}</div>', unsafe_allow_html=True)

# Display chat history in the main section
st.write("Chat History:")
for chat in st.session_state.chat_session:
    if chat["role"] == "user":
        st.markdown(f'<div class="message user-message">You: {chat["context"]}</div>', unsafe_allow_html=True)
    elif chat["role"] == "assistant":
        st.markdown(f'<div class="message bot-message">Bot: {chat["context"]}</div>', unsafe_allow_html=True)

# Button to clear chat history
if st.button("Clear History"):
    clear_history()
    st.success("Chat history has been cleared.")
