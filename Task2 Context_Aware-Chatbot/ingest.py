from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


SAMPLE_CORPUS = [
    {
        "title": "Artificial Intelligence",
        "content": """
        Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to
        natural intelligence displayed by animals including humans. AI research has been defined
        as the field of study of intelligent agents, which refers to any system that perceives
        its environment and takes actions that maximize its chance of achieving its goals.
        Machine learning is a subfield of AI that enables systems to learn from data and improve
        their performance without being explicitly programmed. Deep learning uses neural networks
        with many layers to model complex patterns in data.
        """
    },
    {
        "title": "Python Programming",
        "content": """
        Python is a high-level, general-purpose programming language. Its design philosophy
        emphasizes code readability with significant indentation. Python was created by Guido van
        Rossum and first released in 1991. It has since become one of the most popular languages
        in the world, especially in data science, machine learning, and web development.
        Popular frameworks include Django and Flask for web development, NumPy and Pandas for
        data analysis, and TensorFlow and PyTorch for machine learning.
        """
    },
    {
        "title": "Large Language Models",
        "content": """
        A large language model (LLM) is a language model notable for its ability to achieve
        general-purpose language generation and understanding. LLMs acquire these abilities by
        learning statistical relationships from vast amounts of text during training.
        LLMs use the transformer architecture. GPT by OpenAI and BERT by Google are two
        well-known examples. Retrieval-Augmented Generation (RAG) is a technique that enhances
        LLMs by combining them with a retrieval system to fetch relevant documents and generate
        accurate, grounded responses.
        """
    },
    {
        "title": "LangChain Framework",
        "content": """
        LangChain is an open-source framework for building applications powered by large language
        models. It provides tools for chaining LLM calls, integrating external data sources, and
        managing conversational memory. LangChain supports prompt templates, output parsers,
        vector stores, document loaders, and agents. ConversationalRetrievalChain in LangChain
        allows building chatbots that answer questions from documents while maintaining memory
        of past conversation turns.
        """
    }
]


def load_sample_documents():
    return [
        Document(
            page_content=item["content"].strip(),
            metadata={"source": item["title"]}
        )
        for item in SAMPLE_CORPUS
    ]


def load_uploaded_documents(uploaded_files):
    docs = []
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        docs.append(Document(page_content=text, metadata={"source": file.name}))
    return docs


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"[ingest] {len(chunks)} chunks created from {len(documents)} documents.")
    return chunks


def build_vectorstore(chunks):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    print(f"[ingest] FAISS index built with {vectorstore.index.ntotal} vectors.")
    return vectorstore


def ingest_documents(uploaded_files=None, use_sample=True):
    all_docs = []
    if use_sample:
        all_docs.extend(load_sample_documents())
    if uploaded_files:
        all_docs.extend(load_uploaded_documents(uploaded_files))
    if not all_docs:
        raise ValueError("No documents found. Upload files or enable sample corpus.")
    chunks = split_documents(all_docs)
    return build_vectorstore(chunks)