import os, chromadb
from fastmcp import FastMCP
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Global Setup
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = None 

# This is the "Nuclear" way to ensure the path is always right
DB_PATH = "/home/jray/bigdummy_vector_db"

mcp = FastMCP("BigDummy-RAG")
ALLOWED_BASE_DIR = os.path.expanduser("~/projects") # Safeguard directory
db = chromadb.PersistentClient(path=DB_PATH)

@mcp.tool()
def list_indexed_projects() -> str:
    """List all codebases currently in the BigDummy RAG."""
    cols = db.list_collections()
    return "Indexed projects: " + ", ".join([c.name for c in cols]) if cols else "No projects."

@mcp.tool()
def search_project_code(project_name: str, query: str) -> str:
    """Search a specific project for code snippets using RAG."""
    try:
        col = db.get_collection(project_name)
        v_store = ChromaVectorStore(chroma_collection=col)
        index = VectorStoreIndex.from_vector_store(
            v_store, storage_context=StorageContext.from_defaults(vector_store=v_store)
        )
        return str(index.as_query_engine().query(query))
    except Exception as e:
        return f"Error: Project '{project_name}' not found. {str(e)}"

@mcp.tool()
def write_code_file(file_path: str, content: str) -> str:
    """
    Writes or overwrites a file with new code content.
    Use this to apply fixes or create new modules on BigDummy.
    """
    full_path = os.path.abspath(os.path.expanduser(file_path))
    
    # Security: Ensure we are only writing within the allowed projects folder
    if not full_path.startswith(ALLOWED_BASE_DIR):
        return f"Error: Access Denied. You can only write to subdirectories of {ALLOWED_BASE_DIR}"

    try:
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"✅ Successfully wrote to {file_path}"
    except Exception as e:
        return f"❌ Failed to write file: {str(e)}"

@mcp.tool()
def read_server_logs(lines: int = 50) -> str:
    """
    Reads the last N lines of the BigDummy server logs.
    Use this to diagnose connection issues, GPU errors, or RAG failures.
    """
    LOG_FILE = "/home/jray/bigdummy_logs/bigdummy_server.log"
    try:
        if not os.path.exists(LOG_FILE):
            return f"Error: Log file not found at {LOG_FILE}"
            
        # Efficiently read the tail of the file
        with open(LOG_FILE, "r") as f:
            # We use a simple list slice for the last N lines
            content = f.readlines()
            last_lines = content[-lines:]
            return "".join(last_lines)
    except Exception as e:
        return f"❌ Failed to read logs: {str(e)}"

current_project = "default"

@mcp.tool()
def switch_active_project(project_name: str) -> str:
    """
    Tells the RAG bridge which project collection to focus on.
    Use this when switching from 'habit-tracker' to 'web-app'.
    """
    global current_project
    # Verify the collection exists in Chroma first
    collections = [c.name for c in db.list_collections()]
    if project_name in collections:
        current_project = project_name
        return f"✅ Switched RAG focus to: {project_name}"
    else:
        return f"❌ Project '{project_name}' not found. Available: {', '.join(collections)}"

if __name__ == "__main__":
    try:
        # transport="stdio" is required for the SSH bridge to work
        mcp.run(transport="stdio")
    except Exception as e:
        # Log to stderr so it doesn't break the JSON-RPC stream on stdout
        import sys
        print(f"Server Error: {e}", file=sys.stderr)