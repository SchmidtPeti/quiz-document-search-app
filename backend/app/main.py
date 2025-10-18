from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
import os
import re
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global variables with persistence
vectorstore = None
llm = None
PERSIST_DIRECTORY = "./chroma_db"

class Query(BaseModel):
    question: str
    k: int = 5

def parse_hungarian_legal_document(text: str) -> List[Dict[str, str]]:
    """
    Parse Hungarian legal document preserving hierarchical structure:
    - Fejezet (Chapter)
    - Alcím (Subtitle) 
    - § (Section/Paragraph)
    - Bekezdés (Subsection)
    """
    chunks = []
    
    # Track current context
    current_fejezet = "Ismeretlen Fejezet"
    current_alcim = "Nincs alcím"
    current_paragraph = "0"
    
    # Split by lines to process hierarchically
    lines = text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Detect Fejezet (Chapter) - typically in UPPERCASE or with "Fejezet" keyword
        fejezet_match = re.match(r'^([IVX]+\.?\s*[Ff]ejezet|[A-ZÍÉÁŐÚÖÜÓ\s]+FEJEZET)', line, re.IGNORECASE)
        if fejezet_match:
            current_fejezet = line
            current_alcim = "Nincs alcím"  # Reset subtitle
            i += 1
            continue
        
        # Detect Alcím (Subtitle) - numbered sections before paragraphs
        alcim_match = re.match(r'^\s*(\d+)\.\s+([A-ZÍÉÁŐÚÖÜÓ][^\n§]+)$', line)
        if alcim_match and '§' not in line:
            current_alcim = line
            i += 1
            continue
        
        # Detect Paragraph (§)
        paragraph_match = re.match(r'^(\d+)\.\s*§', line)
        if paragraph_match:
            current_paragraph = paragraph_match.group(1)
            
            # Collect the entire paragraph content (including subsections)
            paragraph_content = [line]
            i += 1
            
            # Continue collecting until next § or Fejezet/Alcím
            while i < len(lines):
                next_line = lines[i].strip()
                
                # Stop if we hit a new structure element
                if (re.match(r'^\d+\.\s*§', next_line) or 
                    re.match(r'^([IVX]+\.?\s*[Ff]ejezet|[A-ZÍÉÁŐÚÖÜÓ\s]+FEJEZET)', next_line, re.IGNORECASE) or
                    (re.match(r'^\s*(\d+)\.\s+([A-ZÍÉÁŐÚÖÜÓ][^\n§]+)$', next_line) and '§' not in next_line)):
                    break
                
                paragraph_content.append(next_line)
                i += 1
            
            # Join paragraph content
            full_paragraph = '\n'.join(paragraph_content).strip()
            
            # Split long paragraphs into subsections
            subsections = split_paragraph_by_subsections(full_paragraph)
            
            for idx, subsection_content in enumerate(subsections):
                if subsection_content.strip():
                    chunks.append({
                        "content": subsection_content.strip(),
                        "fejezet": current_fejezet,
                        "alcim": current_alcim,
                        "paragraph": current_paragraph,
                        "subsection": str(idx),
                        "hierarchy": f"{current_fejezet} > {current_alcim} > §{current_paragraph}"
                    })
            continue
        
        i += 1
    
    # Fallback for unstructured content
    if not chunks:
        return fallback_chunking(text)
    
    return chunks

def split_paragraph_by_subsections(paragraph_text: str) -> List[str]:
    """
    Split a paragraph by bekezdés (subsections marked with (1), (2), etc.)
    Keep max 2000 chars per chunk to avoid context overflow
    """
    MAX_CHUNK_SIZE = 2000
    
    # Split by subsection markers like (1), (2), etc.
    subsection_pattern = r'(\([0-9]+\))'
    parts = re.split(subsection_pattern, paragraph_text)
    
    chunks = []
    current_chunk = ""
    
    for part in parts:
        # If adding this part exceeds limit, save current and start new
        if len(current_chunk) + len(part) > MAX_CHUNK_SIZE and current_chunk:
            chunks.append(current_chunk)
            current_chunk = part
        else:
            current_chunk += part
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks if chunks else [paragraph_text]

def fallback_chunking(text: str) -> List[Dict[str, str]]:
    """Fallback chunking when structure detection fails"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    text_chunks = splitter.split_text(text)
    
    return [{
        "content": chunk,
        "fejezet": "Ismeretlen",
        "alcim": "Nincs alcím",
        "paragraph": "0",
        "subsection": "0",
        "hierarchy": "Általános"
    } for chunk in text_chunks]

@app.post("/upload_document/")
async def upload_document(file: UploadFile = File(...)):
    global vectorstore
    try:
        content = await file.read()
        text = content.decode("utf-8")
        
        # Parse with hierarchical structure
        chunks_with_metadata = parse_hungarian_legal_document(text)
        
        # Extract texts and metadata
        texts = [chunk["content"] for chunk in chunks_with_metadata]
        metadatas = [{
            "fejezet": chunk["fejezet"],
            "alcim": chunk["alcim"],
            "paragraph": chunk["paragraph"],
            "subsection": chunk["subsection"],
            "hierarchy": chunk["hierarchy"],
            "source": file.filename
        } for chunk in chunks_with_metadata]
        
        # Use multilingual embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create or update vector store
        if os.path.exists(PERSIST_DIRECTORY):
            vectorstore = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings,
                collection_name="legal-docs"
            )
            vectorstore.add_texts(texts, metadatas=metadatas)
        else:
            vectorstore = Chroma.from_texts(
                texts, 
                embeddings, 
                metadatas=metadatas,
                collection_name="legal-docs",
                persist_directory=PERSIST_DIRECTORY
            )
        
        vectorstore.persist()
        
        # Statistics
        unique_fejezetek = len(set(m["fejezet"] for m in metadatas))
        unique_alcimek = len(set(m["alcim"] for m in metadatas))
        unique_paragraphs = len(set(m["paragraph"] for m in metadatas))
        
        return {
            "message": "Document indexed successfully",
            "chunks_created": len(texts),
            "fejezetek_found": unique_fejezetek,
            "alcimek_found": unique_alcimek,
            "paragraphs_found": unique_paragraphs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query/")
async def query_document(query: Query):
    global llm
    
    if vectorstore is None:
        raise HTTPException(status_code=400, detail="Upload a document first")
    
    if llm is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
        llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4",
            api_key=api_key
        )
    
    # Custom prompt with hierarchy awareness
    prompt_template = """Egy jogi dokumentum szakértőjeként válaszolj a kérdésre a megadott jogszabályi kontextus alapján.

A kontextus hierarchikus struktúrában van megadva (Fejezet > Alcím > §).

Kontextus:
{context}

Kérdés: {question}

FONTOS: Kövess pontosan ezt a struktúrát a válaszadáshoz:

Ha ez egy feleletválasztós kérdés (A, B, C, D opciókkal):

✓ HELYES VÁLASZ: [betűjel]
Rövid indoklás (1-2 mondat): [miért helyes, hivatkozz a pontos §-ra és bekezdésre]

✗ MIÉRT NEM A TÖBBI:
[betűjel]: [rövid indoklás, miért hibás]
[betűjel]: [rövid indoklás, miért hibás]
[betűjel]: [rövid indoklás, miért hibás]

📚 JOGSZABÁLYI HIVATKOZÁS: [Fejezet, Alcím, § és bekezdés]

---

Ha ez egy nyílt kérdés:

VÁLASZ:
[Rövid, tömör válasz 2-3 mondatban]

RÉSZLETES INDOKLÁS:
[Bővebb kifejtés, ha szükséges]

📚 JOGSZABÁLYI HIVATKOZÁS: [Fejezet, Alcím, § és bekezdés]

---

Válasz:"""

    PROMPT = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Enhanced retrieval with metadata filtering
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": query.k,
            "fetch_k": query.k * 3  # Fetch more for better MMR diversity
        }
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    result = qa_chain({"query": query.question})
    
    # Format sources with hierarchy
    sources = []
    for doc in result["source_documents"]:
        source_info = {
            "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
            "hierarchy": doc.metadata.get("hierarchy", "N/A"),
            "fejezet": doc.metadata.get("fejezet", "N/A"),
            "alcim": doc.metadata.get("alcim", "N/A"),
            "paragraph": doc.metadata.get("paragraph", "N/A"),
            "source_file": doc.metadata.get("source", "N/A")
        }
        sources.append(source_info)
    
    return {
        "answer": result["result"],
        "sources": sources,
        "num_sources": len(sources)
    }

@app.delete("/clear_index/")
async def clear_index():
    global vectorstore
    vectorstore = None
    
    if os.path.exists(PERSIST_DIRECTORY):
        import shutil
        shutil.rmtree(PERSIST_DIRECTORY)
    
    return {"message": "Index cleared successfully"}

@app.get("/health/")
async def health_check():
    return {
        "status": "healthy",
        "index_loaded": vectorstore is not None,
        "llm_loaded": llm is not None
    }