import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize local models
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = Ollama(model="llama3")

# Helper function: extracts plain text from ChromaDB's document objects
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@cl.on_chat_start
async def on_chat_start():
    # 2. Wait for the file upload
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF document (e.g., HR Policy) to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`... Building your local vector database.")
    await msg.send()

    # --- THE RAG PIPELINE ---
    
    # A. Read and Chunk
    loader = PyPDFLoader(file.path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # B. Store chunks in ChromaDB
    vector_db = Chroma.from_documents(documents=texts, embedding=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    # C. Set the Prompt Rules
    prompt_template = """Use the following pieces of context to answer the user's question. 
    If you don't know the answer based on the context, just say that you don't know. Do not make things up.
    If the user asks in Roman Urdu (e.g., "Mujhe kitni leaves milti hain?"), reply naturally in Roman Urdu based ONLY on the context.

    Context: {context}
    Question: {question}

    Helpful Answer:"""
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # D. Build the Modern LCEL Chain (Bypasses the broken module entirely!)
    # This pipes the data directly: Retriever -> Prompt -> LLM -> Output Text
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Save the pipeline to the user's session
    cl.user_session.set("chain", chain)

    msg.content = f"`{file.name}` processed successfully! The document is securely loaded into ChromaDB. Ask a question!"
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the LCEL pipeline
    chain = cl.user_session.get("chain")
    
    res = cl.Message(content="")
    await res.send()

    # Pass the user's question through the pipeline
    response = await chain.ainvoke(
        message.content, 
        config={"callbacks": [cl.AsyncLangchainCallbackHandler()]}
    )

    # LCEL directly outputs the final text string, so we just attach it
    res.content = response
    await res.update()