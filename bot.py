import asyncio
import os
from pprint import pprint
import time
from dotenv import find_dotenv, load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List

if "BASE_DIR" not in st.session_state:
    BASE_DIR = os.path.dirname(__file__)
    os.makedirs(os.path.join(BASE_DIR, "files/"), exist_ok=True)
    st.session_state["BASE_DIR"] = BASE_DIR
else:
    BASE_DIR = st.session_state["BASE_DIR"]


# Define the API keys configuration function
def api_keys_config():
    try:
        if "GOOGLE_API_KEY" not in os.environ:
            # os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
            load_dotenv(find_dotenv(), override=True)
            st.success("GOOGLE_API_KEY set")
        else:
            st.success("GOOGLE_API_KEY is ALready set")
    except Exception as e:
        print(e)
        import getpass

        os.environ["GOOGLE_API_KEY"] = getpass.getpass("GOOGLE_API_KEY")


# LOAD DOCS
def load_docs_locally(files: List[str] = []):
    global BASE_DIR
    from pprint import pprint
    import os

    os.chdir(os.path.join(BASE_DIR, "files/"))
    print(f"current directory: {os.getcwd()}")
    files = [file for file in os.listdir()] if not files else files
    pprint(files)

    data = []

    for file in files:
        _, extension = os.path.splitext(file)
        if not file.startswith("."):
            match extension:
                case ".pdf":
                    from langchain.document_loaders import PyPDFLoader

                    loader = PyPDFLoader(os.path.join(BASE_DIR, "files", file))
                    print(f"loading pdf {os.path.join(BASE_DIR,file)} ....")
                case ".txt":
                    from langchain.document_loaders import TextLoader

                    loader = TextLoader(
                        os.path.join(BASE_DIR, "files", file), encoding="utf-8"
                    )
                    print(f"loading text {os.path.join(BASE_DIR,file)} ....")
                case ".docx":
                    from langchain.document_loaders import Docx2textLoader

                    loader = Docx2textLoader(os.path.join(BASE_DIR, "files", file))
                    print(f"loading docx {os.path.join(BASE_DIR,file)} ....")
                case _:
                    print(f"no such available format such as {extension}")

        data += loader.load()
    os.chdir(BASE_DIR)
    return data


def chunk_data(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text = "\n".join([doc.page_content for doc in docs])
    # print(text)
    chunks = text_splitter.split_text(text)
    return chunks


# Define the vector storage configuration function
def create_index(chunks: List[str]):
    from langchain.vectorstores.chroma import Chroma

    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    pprint("embedding:")
    st.write(embedding)
    # print("embedding.embed_query(chunks[0]):")
    # pprint(embedding.embed_query(chunks[0]))
    # pprint("chunks:")
    pprint(type(chunks))

    vector_store = Chroma.from_texts(chunks, embedding)
    pprint(vector_store)
    return vector_store


from typing import List


def load_docs(docs_urls=["https://pypi.org/"]):
    from langchain.document_loaders.async_html import AsyncHtmlLoader

    print("loading started....")
    loader = AsyncHtmlLoader(docs_urls)
    documents = loader.load()
    return documents


def clean_html(html_page: str, title: str):
    from pprint import pprint
    from bs4 import BeautifulSoup

    parser = BeautifulSoup(html_page, "html.parser")
    # pprint(parser.prettify())
    with open(f"files/{title}.txt", "w", encoding="utf-8") as f:
        for string in parser.strings:
            if string != "\n":
                f.write(string.strip())
                f.write("\n")


def mass_download(urls: List[str]):
    file_titles = []
    html_pages = load_docs(urls)
    for i, html_page in enumerate(html_pages):
        cleaned_file_title = (
            urls[i]
            .replace("/", "_")
            .replace(".", "_")
            .replace("-", "_")
            .replace("https:", "")
            .replace("dz", "")
            .replace("net", "")
            .replace("com", "")
            .replace("org", "")
            .replace("edu", "")
            .strip("_")
        )
        clean_html(html_page.page_content, cleaned_file_title)
        file_titles.append(cleaned_file_title)
    return file_titles


# to PREVENT BREAKING THE MAIN LOOP
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return asyncio.get_event_loop()


def ask_question(query, vector_store):

    from langchain.prompts import PromptTemplate

    template = """
    use the following pieces of context to answer the question at the end. if you don't the answer just say that you don't know the answer, don't try to make up an answer, keep the answer as concise as possible
    {context}
    Question:{question}
    """
    QA_CHAIN_TEMPLATE = PromptTemplate.from_template(template)
    chroma_chain = RetrievalQA.from_chain_type(
        llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=1),  # type: ignore
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_TEMPLATE},
        verbose=True,
    )

    response = chroma_chain({"query": query})
    return response


# Define the question answering function
def searching_with_custom_prompt(query, vector_store, search_type="llm"):
    from langchain.chains import ConversationalRetrievalChain
    from langchain_google_genai import GoogleGenerativeAI
    from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
    from langchain.prompts import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=FileChatMessageHistory("chat_history.json"),
        input_key="question",
        output_key="answer",
    )

    system_message_prompt = """
  use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
  Context: ```{context}```
  """

    user_message_prompt = """
  Question: ```{question}```
  Chat History: ```{chat_history}```
  """

    messages = [
        SystemMessagePromptTemplate.from_template(system_message_prompt),
        HumanMessagePromptTemplate.from_template(user_message_prompt),
    ]

    qa_prompt = ChatPromptTemplate.from_messages(messages)
    get_or_create_eventloop()

    llm = GoogleGenerativeAI(model="gemini-pro")  # type: ignore
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=True,
    )
    return chain.invoke({"question": query})


# Define a thread to perform retrieval augmented generation in the background
def generate_response_in_background():
    if "vector_store" in st.session_state:
        response = searching_with_custom_prompt(
            question, st.session_state["vector_store"]
        )
        return response
    else:
        st.error("vector store not found.")


if __name__ == "__main__":

    api_keys_config()
    # Create a Streamlit app
    st.title("Retrieval Augmented Generation Chatbot")

    # Create a text area for the user to input their documents
    if "vector_store" not in st.session_state:
        st.header("Upload Documents")
        documents = st.text_area(
            "Paste your documents filepath here, separated by a newline:"
        )
        if st.button("load documents"):
            if documents:
                documents = documents.split("\n")
                urls = [
                    "https://fsciences.univ-setif.dz/main_page/english",
                ]
                mass_download(urls)
                documents = [doc.strip() for doc in documents]
                with st.spinner("loading document and creating index ..."):
                    docs = load_docs_locally(documents)
                    print(len(docs))
                    chunks = chunk_data(docs)
                    print(f"{len(chunks)} chunk")
                    vector_store = create_index(chunks)
                    st.session_state["vector_store"] = vector_store
                    st.success("vector storage created!")
                    if st.button("move to chat"):
                        with st.spinner():
                            time.sleep(0.5)
                            st.session_state["prepared for chat"] = True
    else:
        # Create a text input for the user to input their questions
        st.header("Ask a Question")
        question = st.text_input("Ask a question:")

        # Create a button for the user to submit their question
        st.button("Submit")

        if question:
            response = ""
            vector_store = st.session_state.get("vector_store", None)
            print("vector_store:")
            pprint(vector_store)
            # Call the generate_response function
            with st.spinner("Generating response..."):
                if vector_store:
                    # LLM QUESTION ANSWER CHAIN
                    #############################################################################################
                    response = ask_question(question, vector_store)
                    #############################################################################################

                else:
                    st.error(
                        "Please upload documents first. You can do this by clicking the 'Upload Documents' button., or check if the vector store has been created!"
                    )
                if response:
                    st.markdown(response.get("result", "nothing returned"))
                else:
                    st.warning(
                        "couldn't find any relevant documents. Please try again with a different question."
                    )
