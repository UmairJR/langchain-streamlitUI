import streamlit as st
import re
import nltk

nltk.download('punkt')


st.set_page_config(page_title="Welcome", page_icon="👨‍💻")

# Set up the sidebar
st.sidebar.title("AI Tools")
st.sidebar.markdown("""
    **Tools:**
    - PDFBot
    - URLBot
""")

st.title("Welcome 👋")
st.subheader("🌟 Langchain Tools - Gen AI")


st.markdown("Feel free to explore the tools by clicking the pages on the sidebar.")
st.markdown("""
**⚠️ Please Note: Without entering a valid OpenAI API key, none of the tools will function.**
""", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
Please enter your OpenAI API Key below to proceed. Ensure the key is valid and starts with `sk-`.
""")

def is_valid_openai_key(key):
    return re.match(r"^sk-[a-zA-Z0-9_]{48}$", key) is not None


openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-")

if openai_key:
    if is_valid_openai_key(openai_key):
        st.session_state["OPENAI_API_KEY"] = openai_key
        st.success("Valid Key. You may now proceed.")
    else:
        st.error("Invalid API Key. Please enter a valid key.")
else:
    if "OPENAI_API_KEY" in st.session_state and st.session_state["OPENAI_API_KEY"]:
        st.success("API Key is already set. Please proceed.")
    else:
        st.info("Please enter your API Key above.")


st.markdown("""
---
<div style="text-align: center;">
    <p>🔧 Built by <strong>Umair Jr.</strong></p>
    <p>Curiously made to explore LangChain and Gen AI</p>
    <p>Follow me on <a href="https://github.com/UmairJR" target="_blank"><strong>GitHub</strong></a></p>
</div>
""", unsafe_allow_html=True)
# python -m streamlit run 0_👋_Welcome.py