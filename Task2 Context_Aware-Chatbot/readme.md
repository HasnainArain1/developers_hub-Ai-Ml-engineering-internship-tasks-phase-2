# Task 4: Context-Aware Chatbot Using LangChain & RAG

## Objective
Build a conversational chatbot that retrieves answers from a custom knowledge base and remembers context across multiple conversation turns using Retrieval-Augmented Generation (RAG) and LangChain.

## Tech Stack
- **LLM:** Groq (llama-3.1-8b-instant) — Free
- **Embeddings:** HuggingFace all-MiniLM-L6-v2 (runs locally)
- **Vector Store:** FAISS
- **Framework:** LangChain 1.x
- **Deployment:** Streamlit
- **Language:** Python 3.13

## Approach
Documents are loaded and split into 500-token chunks, then embedded using HuggingFace sentence-transformers and stored in a FAISS vector index. On each user query, the top 3 most relevant chunks are retrieved and passed to the LLM along with conversation history. The chain has two stages — first it rephrases follow-up questions into standalone searchable questions using chat history, then it generates an answer strictly from the retrieved context. If the answer is not in the corpus, the bot responds with "I don't have information about that in my knowledge base."

## How to Run

1. Clone the repository and activate your virtual environment
2. Install dependencies with `pip install -r requirements.txt`
3. Get a free Groq API key at https://console.groq.com
4. Create a `.env` file and add `GROQ_API_KEY=your_key_here`
5. Run with `streamlit run app.py`

## Key Results
The chatbot correctly answers questions from the corpus, handles multi-turn conversations using memory, and rejects out-of-corpus questions without hallucinating. 