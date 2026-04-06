# Logistics Document Intelligence (RAG System) 📦

A Retrieval-Augmented Generation (RAG) system designed for Logistics and Supply Chain document intelligence. 

## Features
* **Document Ingestion**: Upload professional PDFs (reports, manuals, bills of lading).
* **Vector Search**: Uses Vector Embeddings and ChromaDB for semantic retrieval.
* **Grounded AI**: Connected to an LLM with strict prompting to prevent hallucination.
* **Web UI**: Built with Streamlit for a seamless chat experience.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Add your API keys to a `.env` file.
4. Run the app: `streamlit run app.py`