# Local Private RAG Chatbot 🦙

A fully local, privacy-first Retrieval-Augmented Generation (RAG) chatbot designed to process and answer questions about sensitive documents (like HR policies) without sending data to the cloud. 

This project utilizes modern **LangChain Expression Language (LCEL)** for fast pipeline execution and supports multilingual querying, including natural responses in Roman Urdu.

## 🧠 Architecture
- **Frontend:** Chainlit (Async streaming UI)
- **Document Processing:** PyPDFLoader & RecursiveCharacterTextSplitter
- **Vector Store:** ChromaDB (Local embeddings)
- **Embeddings:** `nomic-embed-text` (via Ollama)
- **LLM:** `llama3` / `llama3.2` (via Ollama)

## 🚀 How It Works
1. **Document Upload:** User uploads a PDF.
2. **Chunking & Embedding:** The document is split into 1,000-character chunks and embedded into a local Chroma vector database using `nomic-embed-text`.
3. **Retrieval:** User queries the bot (e.g., *"Mujhe annual leaves kitni milti hain?"*). The system retrieves the top 3 most mathematically relevant document chunks.
4. **Generation:** Llama 3 processes the chunks and streams the final answer directly to the UI.

## 🛠️ Installation & Setup
1. Clone this repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `.\venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Ensure [Ollama](https://ollama.com/) is installed and run:
   - `ollama pull llama3`
   - `ollama pull nomic-embed-text`
6. Start the app: `chainlit run app.py -w`

## 🗺️ Roadmap / Future Features
- [ ] Implement Llama 3.2-Vision for processing scanned images and diagrams.
- [ ] Add multi-document upload support.
- [ ] Include conversational memory so the bot remembers follow-up questions.