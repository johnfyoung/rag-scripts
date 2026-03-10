import sys

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# 1. Global Settings (The "Brain" configuration)
# Using a local embedding model so BigDummy does the work, not the cloud
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
DB_PATH = "/home/jray/bigdummy_vector_db"

# 2. Setup Persistent Storage on BigDummy
# This folder will hold ALL your collections
db = chromadb.PersistentClient(path=DB_PATH)

def index_repository(repo_path, collection_name):
    print(f"--- Indexing {collection_name} ---")
    
    # 3. Create or switch to the specific collection for THIS repo
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Load Data with "Code Intelligence"
    # Recursive=True finds all subfolders
    # exclude ignores the junk you don't want the AI to read
    reader = SimpleDirectoryReader(
        input_dir=repo_path,
        recursive=True,
        exclude=["*.git*", "*node_modules*", "*__pycache__*", "*.venv*"],
        exclude_hidden=True
    )
    
    documents = reader.load_data()

    # 5. Build/Update the Index
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context,
        show_progress=True
    )
    return index

# Example Usage:
# index_repository("/home/user/github/my-web-app", "web_app_repo")
# index_repository("/home/user/github/data-science-tool", "ds_tool_repo")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: ./index_coderepo.py <path_to_repo> <collection_name>")
    else:
        repo_path = sys.argv[1]
        coll_name = sys.argv[2]
        index_repository(repo_path, coll_name)
