import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Import the chunking function we built in Step 2
from document_processor import load_and_chunk_documents

# Load environment variables (.env file for OPENAI_API_KEY)
load_dotenv()

# The local folder where our database will be saved
CHROMA_PATH = "chroma_db"

def get_embedding_model():
    # This runs locally on your machine for 100% free!
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vector_store():
    """Converts document chunks to embeddings and saves them to ChromaDB."""
    chunks = load_and_chunk_documents()
    
    if not chunks:
        print("No document chunks found. Cannot build vector store.")
        return None
        
    print(f"Initializing Embedding Model...")
    embedding_function = get_embedding_model()

    print(f"Creating and populating Chroma Vector DB at '{CHROMA_PATH}'...")
    # This automatically embeds the chunks and saves them to disk
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH
    )
    
    print(f"✅ Successfully embedded {len(chunks)} chunks and saved to database!")
    return db

def get_vector_store():
    """Loads the existing Vector DB from disk for future querying."""
    embedding_function = get_embedding_model()
    if os.path.exists(CHROMA_PATH):
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        return db
    else:
        print(f"⚠️ Vector DB not found at '{CHROMA_PATH}'. Please build it first.")
        return None

# Test the script if run directly
if __name__ == "__main__":
    print("--- Building Logistics Vector Database ---")
    build_vector_store()