from vector_store import get_vector_store

def retrieve_context(query, k=3):
    """
    Searches the Vector DB for the 'k' most relevant chunks to the user's query.
    Returns a list of tuples containing the document chunk and its similarity score.
    """
    db = get_vector_store()
    if not db:
        return None
    
    print(f"\n🔍 Searching database for: '{query}'...")
    
    # Perform similarity search
    # k=3 means we want the top 3 most relevant chunks
    results = db.similarity_search_with_score(query, k=k)
    
    return results

# Test the script
if __name__ == "__main__":
    test_query = "Summarize the document."
    
    results = retrieve_context(test_query)
    
    if results:
        print(f"\n--- Top {len(results)} Retrieved Chunks ---")
        for i, (doc, score) in enumerate(results):
            print(f"\n🟢 Match {i+1} (Distance Score: {score:.4f})")
            print(f"📄 Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')}")
            print(f"📝 Content snippet: {doc.page_content[:250]}...\n")
            print("-" * 50)