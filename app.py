import streamlit as st
import os
from vector_store import build_vector_store
from rag_pipeline import generate_answer

# Ensure data directory exists
if not os.path.exists("data"):
    os.makedirs("data")

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Logistics RAG Assistant", page_icon="📦")
st.title("📦 Logistics Document Intelligence")
st.markdown("Upload your logistics manuals, reports, or bills of lading, and ask grounded questions!")

# --- SIDEBAR: DOCUMENT UPLOAD ---
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader("Upload Logistics PDFs", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Saving and processing documents..."):
                # Save uploaded files to the 'data' directory
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("data", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Rebuild the Vector DB with the new files
                build_vector_store()
                st.success("Documents successfully processed and added to the database!")
        else:
            st.warning("Please upload at least one PDF first.")

# --- MAIN CHAT INTERFACE ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages =[]

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your logistics documents..."):
    # 1. Display user's question
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate and display the assistant's answer
    with st.chat_message("assistant"):
        with st.spinner("Searching logistics documents..."):
            response = generate_answer(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})