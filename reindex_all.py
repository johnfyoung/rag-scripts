import os, chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
DB_PATH = "/home/jray/bigdummy_vector_db"
db = chromadb.PersistentClient(path=DB_PATH)

def refresh_all_collections():
    collections = db.list_collections()
    if not collections:
        print("No collections found to refresh.")
        return

    for col in collections:
        print(f"🔄 Refreshing {col.name}...")
        # Note: In a real setup, you'd store the 'repo_path' in the collection's 
        # metadata during initial indexing. Here we assume a standard pathing 
        # or that you'll pass them in. 
        # For now, this serves as a template to loop through your DB.
        
    print("✅ All collections processed.")

if __name__ == "__main__":
    refresh_all_collections()