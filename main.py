import os

from typing import TypedDict, List

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END, START

class AgentState(TypedDict):
    url: str
    docs: list[Document]
    chunks: list[Document]
    summary: str

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)

def fetch(state:AgentState) -> AgentState:
    print("Fetching article........")
    loader = WebBaseLoader(state["url"])
    docs = loader.load()
    title = docs[0].metadata.get("title", "No title found")
    print(f"   Title: {title}")
    return {"docs": docs}

def chunk(state:AgentState) -> AgentState:
    print("Splitting as chunks........")
    chunks = splitter.split_documents(state["docs"])
    print(f" Created {len(chunks)} chunks")
    return {"chunks": chunks}

def embed_and_save(state:AgentState) -> AgentState:
    print("Embedding and Saving to database.....")
    Chroma.from_documents(
        documents = state["chunks"],
        embedding = embeddings,
        persist_directory = "./articleiq_db"
    )
    print(f" Saved....")
    return{}

def summarize(state:AgentState) -> AgentState:
    print("Asking Groq......")
    contents = []
    for chunk in state["chunks"][:5]:
        contents.append(chunk.page_content)
    gist = "\n\n".join(contents)
    response = llm.invoke([
    SystemMessage(content="You are an intelligent article summariser. Answer based only on the article content provided."),
    HumanMessage(content=f"""Article content:
                 {gist}
                 Give me:
                 1. Summary
                 2. Key points
                 3. Important people or companies mentioned""")
])
    return {"summary": response.content}

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("fetch", fetch)
    graph.add_node("chunks", chunk)
    graph.add_node("embed", embed_and_save)
    graph.add_node("summarize", summarize)

    graph.add_edge(START, "fetch")
    graph.add_edge("fetch", "chunks")
    graph.add_edge("chunks", "embed")
    graph.add_edge("embed","summarize")
    graph.add_edge("summarize", END)

    return graph.compile()

if __name__ == "__main__":
    url = input("Paste article URL: ")

    pipeline = build_graph()
    result = pipeline.invoke({"url": url})

    print("\n" + "=" * 50)
    print(result["summary"])