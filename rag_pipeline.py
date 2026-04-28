from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from retriever import retrieve_context

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

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
    
    results = retrieve_context(query)
    if not results:
        return "No relevant logistics documents found in the database."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    print("\n🤖 Generating answer based on retrieved documents...")
    response = llm.invoke(prompt)
    
    return response.content

if __name__ == "__main__":
    test_query = "What are the standard procedures or guidelines mentioned in the document?"
    
    answer = generate_answer(test_query)
    
    print("\n================== FINAL ANSWER ==================")
    print(answer)
    print("==================================================")