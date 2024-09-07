from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


st.set_page_config(page_title="PDF Tool", page_icon="📄")

st.title("PDFBot: Tool 📄")
# Description for the tool
st.markdown("""
Welcome to PDFBot! This tool helps you extract and analyze information from your PDF documents.

**How it works:**
1. Upload a PDF file in the sidebar.
2. The tool will read the PDF, split it into manageable chunks, and index the text.
3. Ask questions about the content, and get precise answers from the document.

Make sure to provide a valid OpenAI API key to use the tool effectively.
""")

st.sidebar.title("Upload PDF File 📩")

openai_key = ""
status = st.empty()
if "OPENAI_API_KEY" in st.session_state:
    openai_key = st.session_state["OPENAI_API_KEY"]
    status.success("API Key uploaded 🆗"+openai_key)
else:
    status.error("Please enter API Key")

if not openai_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

pdf = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

main_placeholder = st.empty()

if pdf is not None:
    # Read PDF file
    pdf_obj = PdfReader(pdf)
    # Display number of pages in the PDF
    num_pages = len(pdf_obj.pages)
    st.sidebar.write(f"Number of pages: {num_pages}")
    main_placeholder.text("Data Loading...Started...🚴🚴🚴")
    text = ""
    for page in pdf_obj.pages[:50]:  # Limit to first 50 pages to avoid excessive processing
        text += page.extract_text()
    main_placeholder.text("Text Splitting...Started...🔃🔃🔃")
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    # Create vectorstore from text chunks
    main_placeholder.text("Embedding Vector Started Building...🔨🔨🔨‍")
    embedding = OpenAIEmbeddings(openai_api_key = openai_key)
    vectorstore = FAISS.from_texts(chunks, embedding=embedding)

    # Input for querying the PDF content
    query = main_placeholder.text_input("Ask any question about the PDF:")

    if query:
        # Perform similarity search to find relevant chunks
        similar_chunks = vectorstore.similarity_search(query=query, k=2)

        # Initialize LLM
        llm = OpenAI(openai_api_key=openai_key, model_name="gpt-3.5-turbo")
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        # Get the response from the LLM
        response = chain.run(input_documents=similar_chunks, question=query)

        st.header("Answer")
        st.write(response)

        st.subheader("Reference Documents:")
        for i, chunk in enumerate(similar_chunks):
            st.write(f"Document {i + 1}:")
            st.write(chunk)