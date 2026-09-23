import os
import re
from dotenv import load_dotenv
import chainlit as cl
import pytesseract
from PIL import Image
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from chainlit.types import ThreadDict

# Load environment variables from .env
load_dotenv()

# --- Persistence, conversion, compression ---
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from pdf2docx import Converter as Pdf2DocxConverter
from docx2pdf import convert as docx2pdf_convert
from pypdf import PdfReader, PdfWriter

# Point pytesseract to the installed Tesseract executable on your Windows machine
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize models
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- LLM PARAMETERS ---
llm = Ollama(
    model="llama3.2",
    temperature=0.2,       
    repeat_penalty=1.3,    
    num_ctx=8192,          
    num_predict=400,       
)

# --- OUTPUT SANITIZER ---
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

def sanitize_ai_turn(text: str) -> str:
    if DEVANAGARI_RE.search(text):
        return "[response skipped — invalid script]"
    return text.strip()

# --- Chat memory: Cloud PostgreSQL persistence layer (Neon) ---
@cl.data_layer
def get_data_layer():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in .env file.")
    return SQLAlchemyDataLayer(conninfo=db_url, ssl_require=True)

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if username == "user" and password == "password":
        return cl.User(identifier="user")
    return None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- File Converter & Compressor Utilities ---
def convert_pdf_to_docx(input_path: str, output_path: str) -> None:
    converter = Pdf2DocxConverter(input_path)
    converter.convert(output_path)
    converter.close()

def convert_docx_to_pdf(input_path: str, output_path: str) -> None:
    docx2pdf_convert(input_path, output_path)

def compress_pdf(input_path: str, output_path: str, image_quality: int = 40) -> None:
    reader = PdfReader(input_path)
    writer = PdfWriter()
    for page in reader.pages:
        for img in page.images:
            try:
                img.replace(img.image, quality=image_quality)
            except Exception:
                pass
        page.compress_content_streams()
        writer.add_page(page)
    writer.compress_identical_objects()
    with open(output_path, "wb") as f:
        writer.write(f)

def compress_image(input_path: str, output_path: str, quality: int = 60, max_dimension: int = 2000) -> None:
    img = Image.open(input_path)
    img.thumbnail((max_dimension, max_dimension))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(output_path, optimize=True, quality=quality)

# --- SIMPLIFIED PROMPTS ---
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
    Reply in the same language the user just wrote in.
    If they wrote in English, reply in English.
    If they wrote in Roman Urdu (Urdu spelled with English letters), reply the same way.
    Keep answers short and directly on topic.
    
    Context: {context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

standard_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
    Reply in the same language the user just wrote in.
    If they wrote in English, reply in English.
    If they wrote in Roman Urdu (Urdu spelled with English letters), reply the same way.
    Keep answers short and directly on topic."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Hello! I am active. You can chat with me normally, or click the **📎 paperclip icon** to upload a **PDF, DOCX, TXT, or Image (PNG/JPG)** for me to read!").send()
    cl.user_session.set("has_document", False)
    cl.user_session.set("chat_history", [])

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    cl.user_session.set("has_document", False)

    history = []
    for step in thread["steps"]:
        if step["type"] == "user_message":
            history.append(HumanMessage(content=step["output"]))
        elif step["type"] == "assistant_message":
            history.append(AIMessage(content=step["output"]))

    if len(history) > 4:
        history = history[-4:]

    cl.user_session.set("chat_history", history)

@cl.on_message
async def main(message: cl.Message):
    if message.elements:
        file = message.elements[0]
        msg = cl.Message(content=f"Processing `{file.name}`...")
        await msg.send()

        # --- FILE TYPE ROUTER ---
        if file.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(file.path)
            documents = loader.load()
        elif file.name.lower().endswith(".docx"):
            loader = Docx2txtLoader(file.path)
            documents = loader.load()
        elif file.name.lower().endswith(".txt"):
            loader = TextLoader(file.path)
            documents = loader.load()
        elif file.name.lower().endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(file.path)
            extracted_text = pytesseract.image_to_string(image)
            documents = [Document(page_content=extracted_text, metadata={"source": file.name})]
        else:
            msg.content = f"❌ Unsupported file type: `{file.name}`."
            await msg.update()
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        vector_db = Chroma.from_documents(documents=texts, embedding=embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})

        rag_chain = (
            {
                "context": lambda x: format_docs(retriever.invoke(x["question"])),
                "question": lambda x: x["question"],
                "chat_history": lambda _: cl.user_session.get("chat_history", []),
            }
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        
        cl.user_session.set("rag_chain", rag_chain)
        cl.user_session.set("has_document", True)
        
        msg.content = f"✅ `{file.name}` processed successfully! You can now ask questions about it."
        await msg.update()

        actions = [
            cl.Action(name="compress_file", payload={"path": file.path, "name": file.name}, label="🗜️ Compress this file")
        ]
        if file.name.lower().endswith(".pdf"):
            actions.append(cl.Action(name="convert_pdf_to_docx", payload={"path": file.path, "name": file.name}, label="🔄 Convert to DOCX"))
        elif file.name.lower().endswith(".docx"):
            actions.append(cl.Action(name="convert_docx_to_pdf", payload={"path": file.path, "name": file.name}, label="🔄 Convert to PDF"))
        await cl.Message(content="File tools:", actions=actions).send()
        
        if not message.content:
            return

    has_document = cl.user_session.get("has_document")
    history = cl.user_session.get("chat_history", [])

    res = cl.Message(content="")
    await res.send()

    if has_document:
        chain = cl.user_session.get("rag_chain")
        async for chunk in chain.astream(
            {"question": message.content}, 
            config={"callbacks": [cl.AsyncLangchainCallbackHandler()]}
        ):
            await res.stream_token(chunk)
    else:
        chain = standard_prompt | llm | StrOutputParser()
        async for chunk in chain.astream(
            {"question": message.content, "chat_history": history}, 
            config={"callbacks": [cl.AsyncLangchainCallbackHandler()]}
        ):
            await res.stream_token(chunk)

    await res.update()

    history.append(HumanMessage(content=message.content))
    history.append(AIMessage(content=sanitize_ai_turn(res.content)))
    
    if len(history) > 4:
        history = history[-4:]
        
    cl.user_session.set("chat_history", history)

# --- File Converter & Compressor action handlers ---
@cl.action_callback("convert_pdf_to_docx")
async def on_convert_pdf_to_docx(action: cl.Action):
    path, name = action.payload["path"], action.payload["name"]
    output_path = os.path.splitext(path)[0] + ".docx"
    try:
        convert_pdf_to_docx(path, output_path)
        await cl.Message(
            content=f"✅ Converted `{name}` to DOCX.",
            elements=[cl.File(name=os.path.basename(output_path), path=output_path)],
        ).send()
    except Exception as e:
        await cl.Message(content=f"❌ Conversion failed: {e}").send()

@cl.action_callback("convert_docx_to_pdf")
async def on_convert_docx_to_pdf(action: cl.Action):
    path, name = action.payload["path"], action.payload["name"]
    output_path = os.path.splitext(path)[0] + ".pdf"
    try:
        convert_docx_to_pdf(path, output_path)
        await cl.Message(
            content=f"✅ Converted `{name}` to PDF.",
            elements=[cl.File(name=os.path.basename(output_path), path=output_path)],
        ).send()
    except Exception as e:
        await cl.Message(content=f"❌ Conversion failed: {e}").send()

@cl.action_callback("compress_file")
async def on_compress_file(action: cl.Action):
    path, name = action.payload["path"], action.payload["name"]
    ext = os.path.splitext(name)[1].lower()
    output_path = os.path.splitext(path)[0] + f"_compressed{ext}"

    try:
        if ext == ".pdf":
            compress_pdf(path, output_path)
        elif ext in (".png", ".jpg", ".jpeg"):
            compress_image(path, output_path)
        else:
            await cl.Message(content=f"Compression isn't supported for `{ext}` files.").send()
            return

        original_kb = os.path.getsize(path) / 1024
        new_kb = os.path.getsize(output_path) / 1024
        await cl.Message(
            content=f"✅ Compressed `{name}`: {original_kb:.1f} KB → {new_kb:.1f} KB",
            elements=[cl.File(name=os.path.basename(output_path), path=output_path)],
        ).send()
    except Exception as e:
        await cl.Message(content=f"❌ Compression failed: {e}").send()