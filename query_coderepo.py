import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

# 1. Setup the Brain
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = OpenAI(
    model="gpt-4o",
    api_base="http://localhost:8080/v1",
    api_key="not-needed",
    context_window=32768,
    is_chat_model=True,
    timeout=600.0,  # Add this! 10 minutes (600 seconds)
    default_headers={"Connection": "keep-alive"}
)

# 2. Connect to the Existing Database
db = chromadb.PersistentClient(path="../bigdummy_vector_db")
chroma_collection = db.get_collection("habit_tracker_native") # Use the exact name you used before
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 3. Load the Index from Disk (No re-indexing needed!)
index = VectorStoreIndex.from_vector_store(
    vector_store,
)

# 4. Ask a Professional Question
# Tells the retriever to pull the top 10 most relevant code snippets instead of 2
query_engine = index.as_query_engine(streaming=True, similarity_top_k=10)
response = query_engine.query("Based on the code, how is the state managed in this habit tracker? Is it using Redux, Context API, or something else?")

# 5. Print the stream (so you see the 122B model 'thinking' in real-time)
for text in response.response_gen:
    print(text, end="", flush=True)
print("\n")