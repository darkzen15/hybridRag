import io
import json
import os
import re
import time
import uuid
from typing import AsyncGenerator, List, Optional

import docx
import pypdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from neo4j import GraphDatabase
from ollama import Client
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

app = FastAPI(title="Hybrid RAG Streaming Gateway")

# Dynamic internal Docker network hosts
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant-rag")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama-rag:11434")

# Initialize database and model clients
ollama_client = Client(host=OLLAMA_HOST)
qdrant = QdrantClient(host=QDRANT_HOST, port=6333)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", "password123"))

COLLECTION_NAME = "hybrid_docs"


# --- OpenAI Pydantic Schemas ---
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]

class ModelItem(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "custom-hybrid-rag"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelItem]


# --- Document Parsing & Chunking Helpers ---
def extract_text_from_pdf(file_bytes: bytes) -> str:
    pdf_file = io.BytesIO(file_bytes)
    reader = pypdf.PdfReader(pdf_file)
    extracted_pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_pages.append(page_text.strip())
    return "\n\n".join(extracted_pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    docx_file = io.BytesIO(file_bytes)
    doc = docx.Document(docx_file)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# --- Knowledge Graph Triple Extraction ---
def extract_and_store_triples(text_chunk: str) -> int:
    prompt = f"""
    Extract atomic entity relationships from the text as a JSON list of objects.

    RULES:
    1. ALWAYS return a JSON list of objects: [{{"subject": "...", "predicate": "...", "object": "..."}}]
    2. Split compound entities into separate atomic triples (e.g., split "Qdrant and Ollama" into two triples).
    3. Predicates MUST be short uppercase strings (e.g. CONNECTS_WITH, DEPENDS_ON, USES).

    Text: {text_chunk}

    JSON Output Format:
    [
      {{"subject": "EntityA", "predicate": "RELATION_TYPE", "object": "EntityB"}},
      {{"subject": "EntityA", "predicate": "RELATION_TYPE", "object": "EntityC"}}
    ]
    Respond ONLY with raw JSON.
    """
    try:
        res = ollama_client.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            format="json"
        )
        parsed = json.loads(res["message"]["content"])

        # Handle both single dicts and lists gracefully
        if isinstance(parsed, dict):
            triples = [parsed]
        elif isinstance(parsed, list):
            triples = parsed
        else:
            triples = []

        # Validate triple schema structure
        valid_triples = [
            t for t in triples 
            if isinstance(t, dict) and "subject" in t and "predicate" in t and "object" in t
        ]

        if valid_triples:
            cypher = """
            UNWIND $triples AS t
            MERGE (a:Entity {name: t.subject})
            MERGE (b:Entity {name: t.object})
            WITH a, b, t
            CALL apoc.create.relationship(a, t.predicate, {}, b) YIELD rel
            RETURN count(rel)
            """
            with neo4j_driver.session() as session:
                session.run(cypher, triples=valid_triples)
            return len(valid_triples)
    except Exception as e:
        print(f"[Warning] Failed to extract triples: {e}")
    return 0


# --- Hybrid Retrieval Pipeline ---
def fetch_hybrid_context(user_query: str) -> str:
    vector_context = []
    try:
        # 1. Truncate query to 1500 chars to prevent Ollama embedding context overflow (500 error)
        truncated_query = user_query[-1500:] if len(user_query) > 1500 else user_query
        emb_res = ollama_client.embeddings(model="nomic-embed-text", prompt=truncated_query)
        
        # 2. Use qdrant.query_points() instead of deprecated qdrant.search()
        if qdrant.collection_exists(COLLECTION_NAME):
            response = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=emb_res["embedding"],
                limit=3
            )
            vector_context = [
                point.payload.get("text", "") 
                for point in response.points 
                if point.payload and "text" in point.payload
            ]
    except Exception as e:
        print(f"[Warning] Vector search error on {QDRANT_HOST}: {e}")

    graph_context = []
    try:
        keywords = re.findall(r'\b[A-Z][a-zA-Z0-9_-]*\b', user_query)
        if keywords:
            cypher = """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE a.name IN $keywords OR b.name IN $keywords
            RETURN a.name + ' ' + type(r) + ' ' + b.name AS triple
            LIMIT 10
            """
            with neo4j_driver.session() as session:
                records = session.run(cypher, keywords=keywords)
                graph_context = [rec["triple"] for rec in records]
    except Exception as e:
        print(f"[Warning] Graph search error on Neo4j: {e}")

    return f"""
    === KNOWLEDGE GRAPH TRIPLES ===
    {chr(10).join(graph_context) if graph_context else 'No graph facts found.'}

    === UNSTRUCTURED VECTOR CHUNKS ===
    {chr(10).join(vector_context) if vector_context else 'No vector documents found.'}
    """


# --- OpenAI SSE Streaming Generator ---
async def generate_sse_stream(chat_id: str, created_time: int, model: str, formatted_messages: list) -> AsyncGenerator[str, None]:
    initial_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None
        }]
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"

    ollama_stream = ollama_client.chat(
        model="llama3.2",
        messages=formatted_messages,
        stream=True
    )

    for chunk in ollama_stream:
        delta_text = chunk.get("message", {}).get("content", "")
        if delta_text:
            stream_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": delta_text},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(stream_chunk)}\n\n"

    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# --- API Routes ---
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelItem(id="hybrid-rag-llama3.2")])


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    user_query = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), None)
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message provided.")

    fused_context = fetch_hybrid_context(user_query)

    system_prompt = (
        "You are a Hybrid RAG assistant. Answer the user prompt using ONLY "
        "the provided Knowledge Graph Triples and Vector Chunks as context."
    )
    formatted_messages = [
        {"role": "system", "content": f"{system_prompt}\n\nContext:\n{fused_context}"}
    ]
    for msg in request.messages:
        formatted_messages.append({"role": msg.role, "content": msg.content})

    chat_id = f"chatcmpl-{int(time.time())}"
    created_time = int(time.time())

    if request.stream:
        return StreamingResponse(
            generate_sse_stream(chat_id, created_time, request.model, formatted_messages),
            media_type="text/event-stream"
        )

    ollama_res = ollama_client.chat(model="llama3.2", messages=formatted_messages)
    return ChatCompletionResponse(
        id=chat_id,
        created=created_time,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=ollama_res["message"]["content"]),
                finish_reason="stop"
            )
        ]
    )


@app.post("/ingest")
async def ingest_content(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    """Handles PDF, DOCX, TXT, or raw string ingestion into Qdrant & Neo4j."""
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

    raw_text = ""
    filename = ""

    if file:
        filename = file.filename or ""
        file_bytes = await file.read()
        
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            raw_text = extract_text_from_pdf(file_bytes)
        elif ext in [".docx", ".doc"]:
            raw_text = extract_text_from_docx(file_bytes)
        else:
            raw_text = file_bytes.decode("utf-8", errors="ignore")
    elif text:
        raw_text = text
    else:
        raise HTTPException(status_code=400, detail="Must provide either a file or text string.")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Extracted document text is empty or unreadable.")

    text_chunks = chunk_text(raw_text, chunk_size=1000, overlap=100)
    total_triples = 0
    points = []

    for idx, chunk in enumerate(text_chunks):
        chunk_id = str(uuid.uuid4())
        
        # Embed chunk
        emb_res = ollama_client.embeddings(model="nomic-embed-text", prompt=chunk)
        points.append(
            PointStruct(
                id=chunk_id,
                vector=emb_res["embedding"],
                payload={
                    "text": chunk,
                    "source": filename if filename else "direct_text",
                    "chunk_index": idx
                }
            )
        )

        # Extract graph triples
        total_triples += extract_and_store_triples(chunk)

    # Upsert vectors to Qdrant
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    return {
        "status": "success",
        "filename": filename if filename else "raw_text",
        "chunks_processed": len(text_chunks),
        "vector_status": f"Indexed {len(points)} chunks in Qdrant",
        "graph_status": f"Stored {total_triples} relationship triples in Neo4j"
    }