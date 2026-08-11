import io
import json
import os
import re
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional

import docx
import pypdf
import spacy
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

# Initialize spaCy NLP Model for deterministic NER
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

SPACY_LABEL_MAP = {
    "ORG": "ORGANIZATION",
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "PRODUCT": "TECHNOLOGY",
    "EVENT": "EVENT",
    "WORK_OF_ART": "CONCEPT",
    "LAW": "CONCEPT"
}


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


# --- Text Processing & Decoding Helpers ---
def decode_bytes(file_bytes: bytes) -> str:
    """Try multiple text encodings to handle Windows UTF-16, UTF-8 BOM, and Latin-1."""
    for encoding in ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"]:
        try:
            decoded = file_bytes.decode(encoding)
            return decoded.replace("\x00", "").strip()
        except (UnicodeDecodeError, ValueError):
            continue
    return file_bytes.decode("utf-8", errors="ignore").replace("\x00", "").strip()


def format_json_for_rag(json_str: str) -> str:
    """Format JSON structures into clean, chunkable text blocks."""
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            blocks = []
            for item in parsed:
                if isinstance(item, (dict, list)):
                    blocks.append(json.dumps(item, indent=2))
                else:
                    blocks.append(str(item))
            return "\n\n---\n\n".join(blocks)
        elif isinstance(parsed, dict):
            return json.dumps(parsed, indent=2)
    except Exception:
        pass
    return json_str


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


def extract_file_text(filename: str, file_bytes: bytes) -> Optional[str]:
    """Dynamically route and extract text from PDFs, DOCX, JSON, source code, and text files."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in [".docx", ".doc"]:
        return extract_text_from_docx(file_bytes)

    raw_text = decode_bytes(file_bytes)
    if not raw_text:
        return None

    if ext in [".json", ".jsonl"] or filename.endswith(".json"):
        return format_json_for_rag(raw_text)

    return raw_text


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# --- Hybrid spaCy + LLM Graph Relationship Extraction ---
def extract_spacy_entities(text: str) -> List[Dict[str, str]]:
    """Extract deterministic entities using spaCy in CPU milliseconds."""
    doc = nlp(text)
    entities = []
    seen = set()

    for ent in doc.ents:
        name = ent.text.strip()
        name = re.sub(r'^(the|a|an)\s+', '', name, flags=re.IGNORECASE).strip('"\'` ')
        label = SPACY_LABEL_MAP.get(ent.label_, "CONCEPT")

        if len(name) > 1 and name.lower() not in seen:
            seen.add(name.lower())
            entities.append({"name": name, "label": label})

    return entities


def extract_and_store_triples(text_chunk: str, source_doc: str = "unknown", chunk_id: str = "0") -> int:
    """Combines spaCy NER for entity detection with Ollama for relationship discovery."""
    spacy_entities = extract_spacy_entities(text_chunk)
    entity_names = [e["name"] for e in spacy_entities]

    prompt = f"""
    Task: Connect the identified known entities in the text using valid relationships.

    Pre-identified Known Entities:
    {json.dumps(entity_names)}

    Rules:
    1. Only form relationships between actual entities present in the text.
    2. Format output strictly as JSON:
       {{
         "triples": [
           {{"subject": "EntityA", "predicate": "RELATIONSHIP_VERB", "object": "EntityB"}}
         ]
       }}
    3. Predicates MUST be short uppercase strings (e.g., DEPENDS_ON, USES, CONTAINS, CREATED_BY).

    Text:
    {text_chunk}
    """

    try:
        res = ollama_client.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"num_ctx": 4096, "temperature": 0.0}
        )
        parsed = json.loads(res["message"]["content"])
        triples = parsed.get("triples", [])

        type_map = {e["name"].lower(): e["label"] for e in spacy_entities}

        valid_triples = []
        for t in triples:
            sub = str(t.get("subject", "")).strip()
            obj = str(t.get("object", "")).strip()
            pred = str(t.get("predicate", "")).strip().upper().replace(" ", "_")

            if sub and obj and pred and sub != obj:
                valid_triples.append({
                    "subject": sub,
                    "subject_type": type_map.get(sub.lower(), "CONCEPT"),
                    "predicate": pred,
                    "object": obj,
                    "object_type": type_map.get(obj.lower(), "CONCEPT"),
                    "source_doc": source_doc,
                    "chunk_id": chunk_id
                })

        if valid_triples:
            cypher = """
            UNWIND $triples AS t
            CALL apoc.merge.node([t.subject_type, 'Entity'], {name: t.subject}) YIELD node as a
            CALL apoc.merge.node([t.object_type, 'Entity'], {name: t.object}) YIELD node as b
            CALL apoc.create.relationship(a, t.predicate, {source_doc: t.source_doc, chunk_id: t.chunk_id}, b) YIELD rel
            RETURN count(rel)
            """
            with neo4j_driver.session() as session:
                session.run(cypher, triples=valid_triples)
            return len(valid_triples)

    except Exception as e:
        print(f"[Hybrid Extraction Error]: {e}")

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


# --- API Endpoints ---
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """Dynamically query Ollama and filter out embedding-only models."""
    model_items = []
    try:
        response = ollama_client.list()
        raw_models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])

        for m_info in raw_models:
            name = m_info.get("name") if isinstance(m_info, dict) else getattr(m_info, "name", None)
            if name and not any(kw in name.lower() for kw in EMBEDDING_KEYWORDS):
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

            raw_text = extract_file_text(filename, file_bytes)

            if not raw_text or not raw_text.strip():
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
                total_triples += extract_and_store_triples(
                    text_chunk=chunk,
                    source_doc=filename,
                    chunk_id=chunk_id
                )

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
            raise HTTPException(status_code=400, detail="Text payload is empty.")

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
            total_triples += extract_and_store_triples(
                text_chunk=chunk,
                source_doc="direct_text",
                chunk_id=chunk_id
            )

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