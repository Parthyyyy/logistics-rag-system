import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Directory where logistics documents are stored
DATA_DIR = "data"

def load_and_chunk_documents():
    """Loads PDFs from the data directory and splits them into manageable chunks."""
    
    # 1. Ensure the directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created '{DATA_DIR}' directory. Please place your logistics PDFs inside.")
        return[]

    # 2. Load the PDFs
    print(f"Loading PDFs from '{DATA_DIR}'...")
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()
    
    if not documents:
        print("No PDFs found in the data directory. Please add some to test.")
        return[]
        
    print(f"Successfully loaded {len(documents)} document pages.")

    # 3. Split the text into chunks
    # We use overlapping chunks to ensure sentences/context aren't abruptly cut in half.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # 1000 characters per chunk
        chunk_overlap=200, # 200 character overlap with the previous chunk
        length_function=len
    )

    print("Splitting documents into chunks...")
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    
    return chunks

# Test the script if run directly
if __name__ == "__main__":
    chunks = load_and_chunk_documents()
    if chunks:
        print("\n--- Sample Chunk ---")
        print(f"Source: {chunks[0].metadata['source']}, Page: {chunks[0].metadata['page']}")
        print(chunks[0].page_content[:300] + "...\n")