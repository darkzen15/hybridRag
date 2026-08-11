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

# Environment variables
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant-rag")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama-rag:11434")

# Initialize database & Ollama clients
ollama_client = Client(host=OLLAMA_HOST)
qdrant = QdrantClient(host=QDRANT_HOST, port=6333)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", "password123"))

COLLECTION_NAME = "hybrid_docs"
EMBEDDING_KEYWORDS = ["embed", "nomic-embed", "bge-m3", "minilm", "mxbai"]


# --- Schemas ---
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
    owned_by: str = "ollama"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelItem]


# --- File Extractor Helpers ---
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


# --- Graph Relationship Extraction ---
def extract_and_store_triples(text_chunk: str) -> int:
    prompt = f"""
    Extract atomic entity relationships from the text as a JSON list of objects.

    RULES:
    1. ALWAYS return a JSON list of objects: [{{"subject": "...", "predicate": "...", "object": "..."}}]
    2. Split compound entities into separate atomic triples.
    3. Predicates MUST be short uppercase strings (e.g., CONNECTS_WITH, DEPENDS_ON, USES).

    Text: {text_chunk}

    Respond ONLY with raw JSON.
    """
    try:
        res = ollama_client.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"num_ctx": 4096}
        )
        parsed = json.loads(res["message"]["content"])

        if isinstance(parsed, dict):
            triples = [parsed]
        elif isinstance(parsed, list):
            triples = parsed
        else:
            triples = []

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


# --- Hybrid Retrieval ---
def fetch_hybrid_context(user_query: str) -> str:
    vector_context = []
    try:
        truncated_query = user_query[-1500:] if len(user_query) > 1500 else user_query
        emb_res = ollama_client.embeddings(model="nomic-embed-text", prompt=truncated_query)
        
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
        print(f"[Warning] Vector search error: {e}")

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
        print(f"[Warning] Graph search error: {e}")

    return f"""
    === KNOWLEDGE GRAPH TRIPLES ===
    {chr(10).join(graph_context) if graph_context else 'No graph facts found.'}

    === UNSTRUCTURED VECTOR CHUNKS ===
    {chr(10).join(vector_context) if vector_context else 'No vector documents found.'}
    """


# --- SSE Streaming Generator ---
async def generate_sse_stream(chat_id: str, created_time: int, model_name: str, formatted_messages: list) -> AsyncGenerator[str, None]:
    initial_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(initial_chunk)}\n\n"

    ollama_stream = ollama_client.chat(
        model=model_name,
        messages=formatted_messages,
        stream=True,
        options={"num_ctx": 8192}
    )

    for chunk in ollama_stream:
        delta_text = chunk.get("message", {}).get("content", "")
        if delta_text:
            stream_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": delta_text}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(stream_chunk)}\n\n"

    final_chunk = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"


# --- Dynamic Endpoints ---
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """Dynamically query Ollama and filter out embedding-only models."""
    model_items = []
    try:
        response = ollama_client.list()
        raw_models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])

        for m_info in raw_models:
            name = m_info.get("name") if isinstance(m_info, dict) else getattr(m_info, "name", None)
            if name:
                # Exclude embedding models from the chat drop-down
                if not any(kw in name.lower() for kw in EMBEDDING_KEYWORDS):
                    model_items.append(ModelItem(id=name, object="model", owned_by="ollama"))

    except Exception as e:
        print(f"[Warning] Could not list models from Ollama: {e}")

    if not model_items:
        model_items.append(ModelItem(id="llama3.2:latest", object="model", owned_by="custom-hybrid-rag"))

    return ModelList(data=model_items)


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
    formatted_messages = [{"role": "system", "content": f"{system_prompt}\n\nContext:\n{fused_context}"}]
    for msg in request.messages:
        formatted_messages.append({"role": msg.role, "content": msg.content})

    target_model = request.model if request.model else "llama3.2"
    chat_id = f"chatcmpl-{int(time.time())}"
    created_time = int(time.time())

    if request.stream:
        return StreamingResponse(
            generate_sse_stream(chat_id, created_time, target_model, formatted_messages),
            media_type="text/event-stream"
        )

    ollama_res = ollama_client.chat(
        model=target_model,
        messages=formatted_messages,
        options={"num_ctx": 8192}
    )
    return ChatCompletionResponse(
        id=chat_id,
        created=created_time,
        model=target_model,
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
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

    upload_queue: List[UploadFile] = []
    if files:
        upload_queue.extend(files)
    if file:
        upload_queue.append(file)

    total_chunks = 0
    total_triples = 0
    processed_files = []

    if upload_queue:
        for uploaded_file in upload_queue:
            filename = uploaded_file.filename or "uploaded_doc"
            file_bytes = await uploaded_file.read()
            ext = os.path.splitext(filename)[1].lower()

            if ext == ".pdf":
                raw_text = extract_text_from_pdf(file_bytes)
            elif ext in [".docx", ".doc"]:
                raw_text = extract_text_from_docx(file_bytes)
            elif ext in [".txt", ".md"]:
                raw_text = file_bytes.decode("utf-8", errors="ignore")
            else:
                continue

            if not raw_text.strip():
                continue

            chunks = chunk_text(raw_text, chunk_size=1000, overlap=100)
            points = []

            for idx, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                emb_res = ollama_client.embeddings(model="nomic-embed-text", prompt=chunk)
                
                points.append(
                    PointStruct(
                        id=chunk_id,
                        vector=emb_res["embedding"],
                        payload={"text": chunk, "source": filename, "chunk_index": idx}
                    )
                )
                total_triples += extract_and_store_triples(chunk)

            if points:
                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                total_chunks += len(chunks)
                processed_files.append(filename)

        return {
            "status": "success",
            "files_processed": processed_files,
            "total_files": len(processed_files),
            "chunks_processed": total_chunks,
            "vector_status": f"Indexed {total_chunks} chunks in Qdrant",
            "graph_status": f"Stored {total_triples} relationship triples in Neo4j"
        }

    elif text:
        raw_text = text.strip()
        if not raw_text:
            raise HTTPException(status_code=400, detail="Text string is empty.")

        chunks = chunk_text(raw_text, chunk_size=1000, overlap=100)
        points = []

        for idx, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            emb_res = ollama_client.embeddings(model="nomic-embed-text", prompt=chunk)
            points.append(
                PointStruct(
                    id=chunk_id,
                    vector=emb_res["embedding"],
                    payload={"text": chunk, "source": "direct_text", "chunk_index": idx}
                )
            )
            total_triples += extract_and_store_triples(chunk)

        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        return {
            "status": "success",
            "filename": "raw_text",
            "chunks_processed": len(chunks),
            "vector_status": f"Indexed {len(chunks)} chunks in Qdrant",
            "graph_status": f"Stored {total_triples} relationship triples in Neo4j"
        }

    else:
        raise HTTPException(status_code=400, detail="Must provide files or a text payload.")