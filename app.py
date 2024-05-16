import streamlit as st
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
import os
import getpass
st.set_page_config(page_title="Boddy Bot", page_icon=":robot:")
def api_keys_config():
  try:
    if "GOOGLE_API_KEY" not in os.environ:
      load_dotenv(find_dotenv(), override=True)
      st.success("GOOGLE_API_KEY set")
    else:
      st.warning("GOOGLE_API_KEY not set")
  except Exception as e:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("GOOGLE_API_KEY")
def insert_or_create_index(index_name, chunks):
    import pinecone
    from pinecone import PodSpec
    from langchain_community.vectorstores.pinecone import Pinecone
    from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
    pc = pinecone.Pinecone()
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    if index_name in pc.list_indexes().names():
        print(f"start fetching from {index_name}!")
        vector_store = Pinecone.from_existing_index(index_name, embedding)
        print(f"done fetching from {index_name}!")
    else:
        print(f"start creating from {index_name}!")
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=PodSpec(environment="gcp-starter"),
        )
        vector_store = Pinecone.from_texts(chunks, embedding, index_name=index_name)
        print(f"done creation of {index_name}!")
    return vector_store

def delete_index(index_name="all"):
    from pinecone import Pinecone

    pc = Pinecone()
    if index_name == "all":
        for index in pc.list_indexes().names():
            pc.delete_index(index)
    else:
        pc.delete_index(index_name)
    print(f"deleted {index_name}")

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
    llm = GoogleGenerativeAI(model="gemini-pro")
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=True,
    )
    return chain.invoke({"question": query})


if __name__ == "__main__":
  api_keys_config()
  st.subheader("Buddy Bot")
  with st.sidebar:
    st.subheader("Settings")
    if "GOOGLE_API_KEY" not in os.environ:
      st.warning("GOOGLE_API_KEY not set")
      google_api_key = st.text_input("GOOGLE_API_KEY")
      if google_api_key:
        st.session_state["GOOGLE_API_KEY"] = google_api_key
        os.environ["GOOGLE_API_KEY"] = st.session_state["GOOGLE_API_KEY"]
        st.success("GOOGLE_API_KEY set")
        if "PINECONE_API_KEY" not in os.environ:
          pinecone_api_key = st.text_input("PINECONE_API_KEY")
          if pinecone_api_key:
            st.session_state["PINECONE_API_KEY"] = pinecone_api_key
            os.environ["PINECONE_API_KEY"] = st.session_state["PINECONE_API_KEY"]
            st.success("PINECONE_API_KEY set")
          st.warning("PINECONE_API_KEY not set")

    document = st.file_uploader("Upload your documents", type=["txt", "pdf"])
    add_button = st.button("Add")
    if document and add_button:
      document_filepath = os.path.join(os.getcwd(), document.name)
      with st.spinner("Saving..."):
        with open(document_filepath, "wb") as f:
          f.write(document.getvalue())
        st.success("Saved")
      with st.spinner("Loading..."):
        from langchain.document_loaders import PyPDFLoader, TextLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        if document.type == "application/pdf":
          loader = PyPDFLoader(document_filepath)
        else:
          loader = TextLoader(document_filepath)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text = "\n".join([doc.page_content for doc in docs])
        chunks = text_splitter.split_text(text)
        st.success(f"Loaded {len(chunks)} chunks")
        delete_index()
        vector_store = insert_or_create_index("test-index", chunks)
        st.session_state["vector_store"] = vector_store
        st.success("Index created")

  query = st.text_input("Ask your question")
  if query and "vector_store" in st.session_state:
    with st.spinner("Loading..."):
      result = searching_with_custom_prompt(query, st.session_state["vector_store"], search_type="llm")
      if not st.session_state.get("chat_history"):
        st.session_state["chat_history"] = []
      # st.session_state["chat_history"].append(result + "\n" + "--"*50 + "\n")
      st.session_state["chat_history"].append(str(result) + "\n" + "--"*50 + "\n")
      answer_text_area = st.empty()
      for chat in st.session_state["chat_history"]:
          answer_text_area.write(chat)