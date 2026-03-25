import streamlit as st
from rag_pipeline import build_rag_chain, get_answer
from ingest import ingest_documents

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("Context-Aware RAG Chatbot")
st.markdown("Ask anything from the loaded knowledge base. The bot remembers your conversation!")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

with st.sidebar:
    st.header("Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload your own .txt documents (optional)",
        type=["txt"],
        accept_multiple_files=True
    )
    use_sample = st.checkbox("Use built-in sample corpus", value=True)

    if st.button("Load & Index Documents"):
        with st.spinner("Indexing... this may take a moment"):
            try:
                vectorstore = ingest_documents(
                    uploaded_files=uploaded_files,
                    use_sample=use_sample
                )
                st.session_state.rag_chain = build_rag_chain(vectorstore)
                st.session_state.documents_loaded = True
                st.success("Ready!")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.documents_loaded:
        st.info("Knowledge base is active.")
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

if not st.session_state.documents_loaded:
    st.warning("Load documents from the sidebar to start chatting.")
else:
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)

    user_input = st.chat_input("Ask a question...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(
                    chain=st.session_state.rag_chain,
                    question=user_input,
                    chat_history=st.session_state.chat_history
                )
            st.write(answer)

        st.session_state.chat_history.append((user_input, answer))
