import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise EnvironmentError(
            print("key not found")
        )

    print("[pipeline] LLM loaded — Groq (llama3-8b-8192)")

    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0,
        max_tokens=512
    )


def build_rag_chain(vectorstore):
    llm = get_llm()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    rephrase_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("human",
         "Given the conversation history above, rephrase the latest question "
         "into a clear standalone question that can be understood without "
         "reading the conversation history. Only rephrase, do not answer.")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, rephrase_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a helpful assistant. Answer the user's question using
ONLY the information provided in the context below.

Rules:
- If the answer is present in the context, answer it accurately and concisely.
- If the answer is NOT in the context, respond with exactly:
  "I don't have information about that in my knowledge base."
- Do NOT use outside knowledge.
- Do NOT make anything up.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    print("[pipeline] RAG chain ready.")
    return chain


def get_answer(chain, question, chat_history):
    formatted_history = []
    for q, a in chat_history:
        formatted_history.append(HumanMessage(content=q))
        formatted_history.append(AIMessage(content=a))

    result = chain.invoke({
        "input": question,
        "chat_history": formatted_history
    })

    answer = result.get("answer", "Sorry, I could not generate an answer.")

    source_docs = result.get("context", [])
    if source_docs:
        sources = set(doc.metadata.get("source", "Unknown") for doc in source_docs)
        print(f"[pipeline] Sources used: {sources}")
    else:
        print("[pipeline] No relevant sources found in knowledge base.")

    return answer