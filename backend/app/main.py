from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# CORS for frontend
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global variables for index (in-memory for simplicity; persist to disk in production)
vectorstore = None
llm = None

class Query(BaseModel):
    question: str

@app.post("/upload_document/")
async def upload_document(file: UploadFile = File(...)):
    global vectorstore
    try:
        content = await file.read()
        text = content.decode("utf-8")

        # Split into chunks (e.g., 1000 chars per chunk for 120-page doc)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)

        # Embed and index
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_texts(chunks, embeddings, collection_name="quiz-docs")
        return {"message": "Document indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def query_document(query: Query):
    global llm
    if vectorstore is None:
        raise HTTPException(status_code=400, detail="Upload a document first")
    if llm is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
        llm = OpenAI(temperature=0.2, api_key=api_key)
    
    # RAG chain: Retrieve top 3 chunks
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    # Custom prompt for answer and references
    prompt = f"Válaszolj a kérdésre a megadott kontextus alapján. Add meg a választ és hivatkozz a releváns bekezdésekre.\nKérdés: {query.question}\nKörnyezet: {{context}}"
    result = qa_chain({"query": query.question, "prompt": prompt})
    
    return {
        "answer": result["result"],
        "sources": [doc.page_content for doc in result["source_documents"]]
    }