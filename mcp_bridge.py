import os
import chromadb
from fastmcp import FastMCP
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Global Setup (Match your indexer)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None 

mcp = FastMCP("BigDummy-RAG")
DB_PATH = os.path.expanduser("~/cac-rag-projects/bigdummy_vector_db")
db = chromadb.PersistentClient(path=DB_PATH)

@mcp.tool()
def list_indexed_projects() -> str:
    """Returns a list of all codebases currently indexed in the RAG database."""
    collections = db.list_collections()
    if not collections:
        return "No projects indexed yet."
    return "Available projects: " + ", ".join([c.name for c in collections])

@mcp.tool()
def search_project_code(project_name: str, query: str) -> str:
    """
    Search a specific indexed project for code logic.
    Use 'list_indexed_projects' first to see available project_names.
    """
    try:
        # Dynamically connect to the requested collection
        chroma_collection = db.get_collection(project_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=StorageContext.from_defaults(vector_store=vector_store)
        )
        query_engine = index.as_query_engine()
        
        response = query_engine.query(query)
        return f"Results from {project_name}:\n\n{str(response)}"
    except Exception as e:
        return f"Error: Project '{project_name}' not found or search failed. {str(e)}"

if __name__ == "__main__":
    mcp.run(transport="stdio")