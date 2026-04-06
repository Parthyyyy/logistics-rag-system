from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from retriever import retrieve_context

# 1. Initialize the LLM
# We use temperature=0 to make the AI strict, factual, and prevent hallucination
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 2. Define a strict System Prompt
PROMPT_TEMPLATE = """
You are a highly professional Logistics and Supply Chain Assistant. 
Answer the user's question using ONLY the provided CONTEXT from official documents.
If the answer cannot be found in the CONTEXT, do not guess or use outside knowledge. 
Simply reply: "I'm sorry, but I cannot find the answer to this in the provided logistics documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

def generate_answer(query):
    """Retrieves context and generates an answer strictly based on the logistics documents."""
    
    # Step A: Retrieve relevant documents
    results = retrieve_context(query)
    if not results:
        return "No relevant logistics documents found in the database."

    # Step B: Format the context into a single block of text
    # We join the text of the top chunks together, separated by dashed lines
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Step C: Construct the final prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    # Step D: Generate the response
    print("\n🤖 Generating answer based on retrieved documents...")
    response = llm.invoke(prompt)
    
    return response.content

# Test the fully integrated pipeline
if __name__ == "__main__":
    # Change this to a question relevant to your PDF!
    test_query = "What are the standard procedures or guidelines mentioned in the document?"
    
    answer = generate_answer(test_query)
    
    print("\n================== FINAL ANSWER ==================")
    print(answer)
    print("==================================================")