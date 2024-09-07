import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os



st.set_page_config(page_title="URL Tool", page_icon="🔗")

st.title("URLBot: Tool 🔗")
st.markdown("""
Welcome to URLBot! This tool allows you to process and analyze news articles from provided URLs. 

**How it works:**
1. Enter up to 3 URLs of news articles in the sidebar.
2. Click 'Process URLs' to load, process, and index the content.
3. Ask questions about the articles, and get summarized answers along with source details.

Ensure you have a valid OpenAI API key to use the tool.
""")
st.sidebar.title("News Article URLs🌐")
openai_key = ''
status = st.empty()
if "OPENAI_API_KEY" in st.session_state:
    openai_key = st.session_state["OPENAI_API_KEY"]
    status.success("API Key uploaded 🆗")
else:
    status.error("Please enter API Key")

urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process Urls", disabled=not openai_key)
file_path = "faiss_store.pkl"

main_placeholder = st.empty()

# Use ChatOpenAI with gpt-3.5-turbo


if not openai_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

if process_url_clicked:
    # Load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()


    # Check if data is loaded
    if not data:
        st.error("No data loaded from the provided URLs.")
    else:
        # Split data
        main_placeholder.text("Text Splitting...Started...✅✅✅")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )

        docs = text_splitter.split_documents(data)

        # Check if documents were split successfully
        if not docs:
            st.error("No documents found after splitting.")
        else:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

            # Generate document embeddings with error handling
            try:
                main_placeholder.text("Embedding Vector Started Building...🔨🔨🔨‍")
                vectorstore_openai = FAISS.from_documents(docs, embeddings)
                time.sleep(2)
                # Save vectorstore to file
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore_openai, f)
            except Exception as e:
                st.error(f"Unexpected Error: {str(e)}")
            query = main_placeholder.text_input("Question: ")
            if query:
                # Check if FAISS store exists
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        vectorstore = pickle.load(f)
                        llm = ChatOpenAI(api_key=openai_key, temperature=0.9, max_tokens=500)
                        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                        result = chain({"question": query}, return_only_outputs=True)

                        st.header("Answer")
                        st.write(result["answer"])

                        # Display sources
                        sources = result.get("sources", "")
                        if sources:
                            st.subheader("Sources: ")
                            sources_list = sources.split("\n")
                            for source in sources_list:
                                st.write(source)
                else:
                    st.error("No FAISS index found. Please process the URLs first.")

