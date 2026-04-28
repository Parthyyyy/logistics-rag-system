import streamlit as st
import os
from vector_store import build_vector_store
from rag_pipeline import generate_answer

if not os.path.exists("data"):
    os.makedirs("data")

st.set_page_config(page_title="Logistics RAG Assistant", page_icon="📦")
st.title("📦 Logistics Document Intelligence")
st.markdown("Upload your logistics manuals, reports, or bills of lading, and ask grounded questions!")

with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader("Upload Logistics PDFs", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Saving and processing documents..."):
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("data", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                build_vector_store()
                st.success("Documents successfully processed and added to the database!")
        else:
            st.warning("Please upload at least one PDF first.")

if "messages" not in st.session_state:
    st.session_state.messages =[]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your logistics documents..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching logistics documents..."):
            response = generate_answer(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})