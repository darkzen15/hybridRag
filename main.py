import ast
import email
from email import policy
from email.parser import BytesParser
import io
import json
import os
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import docx
import pandas as pd
import pypdf
import spacy
import yaml
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

# Initialize spaCy NLP model for deterministic NER
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


# --- OpenAPI Schemas ---
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


# --- Multi-Encoding & Binary Extraction Helpers ---
def decode_bytes(file_bytes: bytes) -> str:
    """Try multiple text encodings to handle Windows UTF-16, UTF-8 BOM, and Latin-1."""
    for encoding in ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"]:
        try:
            decoded = file_bytes.decode(encoding)
            return decoded.replace("\x00", "").strip()
        except (UnicodeDecodeError, ValueError):
            continue
    return file_bytes.decode("utf-8", errors="ignore").replace("\x00", "").strip()


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
    """Dynamically extract text from PDFs, DOCX, source code, and text files."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext in [".docx", ".doc"]:
        return extract_text_from_docx(file_bytes)

    raw_text = decode_bytes(file_bytes)
    return raw_text if raw_text else None


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# =====================================================================
# ENGINE 1: NATIVE STRUCTURED JSON INGESTION ENGINE
# =====================================================================
def ingest_json_to_neo4j(
    json_data: Any,
    parent_node_id: Optional[str] = None,
    parent_rel: str = "CONTAINS",
    source_doc: str = "unknown"
) -> int:
    """Recursively maps structured JSON trees directly into Neo4j without LLM calls."""
    triples_created = 0
    with neo4j_driver.session() as session:
        if isinstance(json_data, dict):
            node_name = (
                json_data.get("name")
                or json_data.get("id")
                or json_data.get("title")
                or json_data.get("service")
            )

            properties = {}
            nested_structures = {}
            for k, v in json_data.items():
                if isinstance(v, (dict, list)):
                    nested_structures[k] = v
                elif v is not None:
                    properties[k] = str(v)

            if not node_name:
                node_name = f"Object_{uuid.uuid4().hex[:8]}"

            cypher_create_node = """
            MERGE (n:Entity:JSONNode {name: $node_name})
            SET n += $properties, n.source_doc = $source_doc
            RETURN id(n)
            """
            session.run(cypher_create_node, node_name=node_name, properties=properties, source_doc=source_doc)

            if parent_node_id:
                cypher_link = """
                MATCH (p:Entity {name: $parent_node_id})
                MATCH (c:Entity {name: $node_name})
                CALL apoc.create.relationship(p, $rel, {source_doc: $source_doc}, c) YIELD rel
                RETURN count(rel)
                """
                res = session.run(
                    cypher_link,
                    parent_node_id=parent_node_id,
                    node_name=node_name,
                    rel=parent_rel,
                    source_doc=source_doc
                )
                triples_created += res.single()[0] if res.peek() else 1

            for key, nested_val in nested_structures.items():
                rel_type = key.upper().replace("-", "_").replace(" ", "_")
                triples_created += ingest_json_to_neo4j(
                    nested_val,
                    parent_node_id=node_name,
                    parent_rel=rel_type,
                    source_doc=source_doc
                )

        elif isinstance(json_data, list):
            for item in json_data:
                triples_created += ingest_json_to_neo4j(
                    item,
                    parent_node_id=parent_node_id,
                    parent_rel="HAS_ITEM",
                    source_doc=source_doc
                )

    return triples_created


def chunk_json_object_level(json_data: Any) -> List[str]:
    """Chunks JSON by object boundaries so vectors never break syntax brackets."""
    chunks = []
    if isinstance(json_data, list):
        for item in json_data:
            chunks.append(json.dumps(item, indent=2))
    elif isinstance(json_data, dict):
        if len(json.dumps(json_data)) > 1500:
            for k, v in json_data.items():
                chunks.append(json.dumps({k: v}, indent=2))
        else:
            chunks.append(json.dumps(json_data, indent=2))
    else:
        chunks.append(str(json_data))
    return chunks


# =====================================================================
# ENGINE 2: NATIVE TABULAR (CSV/TSV/XLSX) INGESTION ENGINE
# =====================================================================
def ingest_tabular_to_neo4j(df: pd.DataFrame, source_doc: str) -> int:
    """Maps DataFrame rows directly to Neo4j nodes and creates categorical connections."""
    df = df.fillna("")
    table_name = os.path.splitext(source_doc)[0]
    records = df.to_dict(orient="records")
    triples_created = 0

    with neo4j_driver.session() as session:
        for idx, row in enumerate(records):
            row_id = (
                row.get("id")
                or row.get("ID")
                or row.get("name")
                or row.get("Name")
                or f"{table_name}_Row_{idx + 1}"
            )
            properties = {str(k).strip(): str(v).strip() for k, v in row.items() if v != ""}

            cypher_row = """
            MERGE (r:Entity:TabularRow {name: $row_id})
            SET r += $properties, r.source_doc = $source_doc, r.table = $table_name
            RETURN id(r)
            """
            session.run(cypher_row, row_id=str(row_id), properties=properties, source_doc=source_doc, table_name=table_name)
            triples_created += 1

            for col, val in row.items():
                col_clean = str(col).strip()
                val_clean = str(val).strip()
                if val_clean and col_clean.lower() in [
                    "category", "department", "type", "status", "group", "role", "location"
                ]:
                    rel_type = f"HAS_{col_clean.upper().replace(' ', '_')}"
                    cypher_link = """
                    MATCH (r:Entity:TabularRow {name: $row_id})
                    MERGE (v:Entity:Category {name: $val_clean})
                    CALL apoc.create.relationship(r, $rel_type, {source_doc: $source_doc}, v) YIELD rel
                    RETURN count(rel)
                    """
                    session.run(cypher_link, row_id=str(row_id), val_clean=val_clean, rel_type=rel_type, source_doc=source_doc)
                    triples_created += 1

    return triples_created


def chunk_tabular_dataframe(df: pd.DataFrame, max_rows_per_chunk: int = 5) -> List[str]:
    """Formats DataFrame rows into self-contained text blocks preserving column headers."""
    df = df.fillna("")
    records = df.to_dict(orient="records")
    chunks = []

    current_block = []
    for idx, row in enumerate(records):
        formatted_fields = [f"{k}: {v}" for k, v in row.items() if str(v).strip() != ""]
        formatted_row = f"--- Record {idx + 1} ---\n" + "\n".join(formatted_fields)
        current_block.append(formatted_row)

        if len(current_block) >= max_rows_per_chunk:
            chunks.append("\n\n".join(current_block))
            current_block = []

    if current_block:
        chunks.append("\n\n".join(current_block))

    return chunks


# =====================================================================
# ENGINE 3: PYTHON AST CODE INGESTION ENGINE (.py)
# =====================================================================
def ingest_python_ast_to_neo4j(code_str: str, source_doc: str) -> int:
    """Parses Python AST to map modules, classes, functions, and imports in Neo4j."""
    triples_created = 0
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return 0

    module_name = os.path.splitext(source_doc)[0]

    with neo4j_driver.session() as session:
        cypher_mod = """
        MERGE (m:Entity:CodeModule {name: $module_name})
        SET m.source_doc = $source_doc
        RETURN id(m)
        """
        session.run(cypher_mod, module_name=module_name, source_doc=source_doc)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                cypher_class = """
                MATCH (m:Entity:CodeModule {name: $module_name})
                MERGE (c:Entity:CodeClass {name: $class_name})
                SET c.source_doc = $source_doc
                MERGE (m)-[r:DEFINES_CLASS]->(c)
                RETURN count(r)
                """
                res = session.run(cypher_class, module_name=module_name, class_name=class_name, source_doc=source_doc)
                triples_created += res.single()[0] if res.peek() else 1

                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_name = f"{class_name}.{child.name}"
                        cypher_method = """
                        MATCH (c:Entity:CodeClass {name: $class_name})
                        MERGE (f:Entity:CodeFunction {name: $method_name})
                        SET f.source_doc = $source_doc, f.is_async = $is_async
                        MERGE (c)-[r:HAS_METHOD]->(f)
                        RETURN count(r)
                        """
                        res = session.run(
                            cypher_method,
                            class_name=class_name,
                            method_name=method_name,
                            source_doc=source_doc,
                            is_async=isinstance(child, ast.AsyncFunctionDef)
                        )
                        triples_created += res.single()[0] if res.peek() else 1

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_name = node.name
                cypher_func = """
                MATCH (m:Entity:CodeModule {name: $module_name})
                MERGE (f:Entity:CodeFunction {name: $func_name})
                SET f.source_doc = $source_doc, f.is_async = $is_async
                MERGE (m)-[r:DEFINES_FUNCTION]->(f)
                RETURN count(r)
                """
                res = session.run(
                    cypher_func,
                    module_name=module_name,
                    func_name=func_name,
                    source_doc=source_doc,
                    is_async=isinstance(node, ast.AsyncFunctionDef)
                )
                triples_created += res.single()[0] if res.peek() else 1

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    cypher_imp = """
                    MATCH (m:Entity:CodeModule {name: $module_name})
                    MERGE (i:Entity:Library {name: $lib_name})
                    MERGE (m)-[r:IMPORTS]->(i)
                    RETURN count(r)
                    """
                    res = session.run(cypher_imp, module_name=module_name, lib_name=alias.name)
                    triples_created += res.single()[0] if res.peek() else 1

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    cypher_imp = """
                    MATCH (m:Entity:CodeModule {name: $module_name})
                    MERGE (i:Entity:Library {name: $lib_name})
                    MERGE (m)-[r:IMPORTS_FROM]->(i)
                    RETURN count(r)
                    """
                    res = session.run(cypher_imp, module_name=module_name, lib_name=node.module)
                    triples_created += res.single()[0] if res.peek() else 1

    return triples_created


def chunk_python_ast(code_str: str) -> List[str]:
    """Extracts classes, functions, and top-level code as atomic code chunks for Qdrant."""
    chunks = []
    try:
        tree = ast.parse(code_str)
        lines = code_str.splitlines()

        header_lines = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start = node.lineno - 1
                end = getattr(node, 'end_lineno', start + 1)
                header_lines.extend(lines[start:end])

        header_prefix = "\n".join(header_lines) + "\n\n" if header_lines else ""

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = getattr(node, 'end_lineno', len(lines))
                body_code = "\n".join(lines[start:end])
                chunks.append(f"# Context Header\n{header_prefix}{body_code}")

    except Exception:
        pass

    if not chunks:
        chunks = chunk_text(code_str, chunk_size=1000, overlap=100)

    return chunks


# =====================================================================
# ENGINE 4: YAML SPEC INGESTION ENGINE (.yaml, .yml)
# =====================================================================
def chunk_yaml_by_sections(yaml_data: Any) -> List[str]:
    """Chunks YAML specs by top-level section keys to preserve context and block hierarchy."""
    chunks = []
    if isinstance(yaml_data, dict):
        for key, val in yaml_data.items():
            section_yaml = yaml.dump({key: val}, default_flow_style=False)
            chunks.append(f"--- Section: {key} ---\n{section_yaml}")
    elif isinstance(yaml_data, list):
        for idx, item in enumerate(yaml_data):
            section_yaml = yaml.dump(item, default_flow_style=False)
            chunks.append(f"--- Item {idx + 1} ---\n{section_yaml}")
    else:
        chunks.append(str(yaml_data))
    return chunks


# =====================================================================
# ENGINE 5: EMAIL INGESTION ENGINE (.eml)
# =====================================================================
def parse_eml_file(file_bytes: bytes) -> dict:
    """Parses .eml file byte contents into structured message metadata and plain text body."""
    msg = BytesParser(policy=policy.default).parsebytes(file_bytes)
    body = ""

    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    body += payload.decode(charset, errors="ignore")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            body = payload.decode(charset, errors="ignore")

    return {
        "subject": str(msg.get("Subject", "No Subject")),
        "from": str(msg.get("From", "Unknown")),
        "to": str(msg.get("To", "Unknown")),
        "date": str(msg.get("Date", "Unknown")),
        "message_id": str(msg.get("Message-ID", f"EML_{uuid.uuid4().hex[:8]}")),
        "in_reply_to": str(msg.get("In-Reply-To", "")) if msg.get("In-Reply-To") else None,
        "body": body.strip()
    }


def ingest_email_to_neo4j(email_data: dict, source_doc: str) -> int:
    """Maps email message topology (sender, recipients, thread replies) directly into Neo4j."""
    triples_created = 0
    msg_id = email_data["message_id"]

    with neo4j_driver.session() as session:
        cypher_email = """
        MERGE (e:Entity:Email {name: $msg_id})
        SET e.subject = $subject, e.date = $date, e.source_doc = $source_doc
        RETURN id(e)
        """
        session.run(
            cypher_email,
            msg_id=msg_id,
            subject=email_data["subject"],
            date=email_data["date"],
            source_doc=source_doc
        )
        triples_created += 1

        if email_data["from"] != "Unknown":
            cypher_from = """
            MATCH (e:Entity:Email {name: $msg_id})
            MERGE (p:Entity:Person {name: $from_person})
            MERGE (p)-[r:SENT]->(e)
            RETURN count(r)
            """
            res = session.run(cypher_from, msg_id=msg_id, from_person=email_data["from"])
            triples_created += res.single()[0] if res.peek() else 1

        if email_data["to"] != "Unknown":
            cypher_to = """
            MATCH (e:Entity:Email {name: $msg_id})
            MERGE (p:Entity:Person {name: $to_person})
            MERGE (e)-[r:SENT_TO]->(p)
            RETURN count(r)
            """
            res = session.run(cypher_to, msg_id=msg_id, to_person=email_data["to"])
            triples_created += res.single()[0] if res.peek() else 1

        if email_data["in_reply_to"]:
            cypher_reply = """
            MATCH (e:Entity:Email {name: $msg_id})
            MERGE (parent:Entity:Email {name: $parent_id})
            MERGE (e)-[r:REPLIED_TO]->(parent)
            RETURN count(r)
            """
            res = session.run(cypher_reply, msg_id=msg_id, parent_id=email_data["in_reply_to"])
            triples_created += res.single()[0] if res.peek() else 1

    return triples_created


def chunk_email_content(email_data: dict) -> List[str]:
    """Combines email header context with body chunks to prevent context loss during vector search."""
    header_summary = (
        f"From: {email_data['from']}\n"
        f"To: {email_data['to']}\n"
        f"Date: {email_data['date']}\n"
        f"Subject: {email_data['subject']}\n"
    )

    if not email_data["body"]:
        return [header_summary]

    body_chunks = chunk_text(email_data["body"], chunk_size=800, overlap=100)
    return [f"{header_summary}\nMessage Content:\n{c}" for c in body_chunks]


# =====================================================================
# ENGINE 6: LOG FILE INGESTION ENGINE (.log)
# =====================================================================
LOG_TIMESTAMP_PATTERN = re.compile(
    r'^(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+'
    r'\[?(?P<level>DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\]?\s+'
    r'(?P<message>.*)',
    re.IGNORECASE
)


def parse_log_entries(log_text: str) -> List[dict]:
    """Groups multi-line log events and stack traces into structured log entry dicts."""
    lines = log_text.splitlines()
    entries = []
    current_entry = None

    for line in lines:
        match = LOG_TIMESTAMP_PATTERN.match(line)
        if match:
            if current_entry:
                entries.append(current_entry)
            current_entry = {
                "timestamp": match.group("timestamp"),
                "level": match.group("level").upper(),
                "message": match.group("message"),
                "details": line
            }
        else:
            if current_entry:
                current_entry["details"] += "\n" + line
            else:
                current_entry = {
                    "timestamp": "UNKNOWN",
                    "level": "INFO",
                    "message": line,
                    "details": line
                }

    if current_entry:
        entries.append(current_entry)

    return entries


def ingest_logs_to_neo4j(log_entries: List[dict], source_doc: str) -> int:
    """Extracts warning/error log events, levels, and raised exception types into Neo4j."""
    triples_created = 0
    file_prefix = os.path.splitext(source_doc)[0]

    with neo4j_driver.session() as session:
        for idx, entry in enumerate(log_entries):
            if entry["level"] in ["ERROR", "CRITICAL", "FATAL", "WARN", "WARNING"]:
                log_id = f"Log_{file_prefix}_{idx + 1}"
                cypher_log = """
                MERGE (l:Entity:LogEvent {name: $log_id})
                SET l.timestamp = $timestamp, l.message = $message, l.level = $level, l.source_doc = $source_doc
                MERGE (lvl:Entity:LogLevel {name: $level})
                MERGE (l)-[r:HAS_LEVEL]->(lvl)
                RETURN count(r)
                """
                res = session.run(
                    cypher_log,
                    log_id=log_id,
                    timestamp=entry["timestamp"],
                    message=entry["message"][:200],
                    level=entry["level"],
                    source_doc=source_doc
                )
                triples_created += res.single()[0] if res.peek() else 1

                error_types = re.findall(r'([A-Za-z0-9_]+Error|[A-Za-z0-9_]+Exception)', entry["details"])
                for err_name in set(error_types):
                    cypher_err = """
                    MATCH (l:Entity:LogEvent {name: $log_id})
                    MERGE (e:Entity:ErrorType {name: $err_name})
                    MERGE (l)-[r:RAISED_ERROR]->(e)
                    RETURN count(r)
                    """
                    res_err = session.run(cypher_err, log_id=log_id, err_name=err_name)
                    triples_created += res_err.single()[0] if res_err.peek() else 1

    return triples_created


def chunk_log_entries(log_entries: List[dict], max_entries_per_chunk: int = 10) -> List[str]:
    """Chunks logs by complete multi-line entries so stack traces are never sliced in half."""
    chunks = []
    current_block = []

    for entry in log_entries:
        current_block.append(entry["details"])
        if len(current_block) >= max_entries_per_chunk:
            chunks.append("\n\n".join(current_block))
            current_block = []

    if current_block:
        chunks.append("\n\n".join(current_block))

    return chunks


# =====================================================================
# ENGINE 7: UNSTRUCTURED PROSE INGESTION ENGINE (spaCy NER + Ollama)
# =====================================================================
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
    """Extracts entity relationships for unstructured text using spaCy NER + Ollama."""
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


# --- Hybrid Context Retrieval ---
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
            ext = os.path.splitext(filename)[1].lower()

            # --- 1. DEDICATED STRUCTURED JSON ROUTE ---
            if ext in [".json", ".jsonl"]:
                try:
                    raw_decoded = decode_bytes(file_bytes)
                    json_payload = json.loads(raw_decoded)

                    total_triples += ingest_json_to_neo4j(json_payload, source_doc=filename)
                    json_chunks = chunk_json_object_level(json_payload)

                    points = []
                    for idx, chunk_str in enumerate(json_chunks):
                        chunk_id = str(uuid.uuid4())
                        emb_res = ollama_client.embeddings(model="nomic-embed-text", prompt=chunk_str)
                        points.append(
                            PointStruct(
                                id=chunk_id,
                                vector=emb_res["embedding"],
                                payload={"text": chunk_str, "source": filename, "chunk_index": idx, "is_json": True}
                            )
                        )

                    if points:
                        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                        total_chunks += len(json_chunks)
                        processed_files.append(filename)

                    continue
                except Exception as e:
                    print(f"[JSON Ingest Error] Falling back to text mode for {filename}: {e}")

            # --- 2. DEDICATED STRUCTURED TABULAR ROUTE (CSV, TSV, XLSX) ---
            if ext in [".csv", ".tsv", ".xlsx", ".xls"]:
                try:
                    if ext == ".csv":
                        df = pd.read_csv(io.BytesIO(file_bytes))
                    elif ext == ".tsv":
                        df = pd.read_csv(io.BytesIO(file_bytes), sep="\t")
                    else:
                        df = pd.read_excel(io.BytesIO(file_bytes))

                    total_triples += ingest_tabular_to_neo4j(df, source_doc=filename)
                    tabular_chunks = chunk_tabular_dataframe(df, max_rows_per_chunk=5)

                    points = []
                    for idx, chunk_str in enumerate(tabular_chunks):
                        chunk_id = str(uuid.uuid4())
                        emb_res = ollama_client.embeddings(model="nomic-embed-text", prompt=chunk_str)
                        points.append(
                            PointStruct(
                                id=chunk_id,
                                vector=emb_res["embedding"],
                                payload={"text": chunk_str, "source": filename, "chunk_index": idx, "is_tabular": True}
                            )
                        )

                    if points:
                        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                        total_chunks += len(tabular_chunks)
                        processed_files.append(filename)

                    continue
                except Exception as e:
                    print(f"[Tabular Ingest Error] Falling back to text mode for {filename}: {e}")

            # --- 3. DEDICATED PYTHON AST CODE ROUTE (.py) ---
            if ext == ".py":
                try:
                    raw_decoded = decode_bytes(file_bytes)
                    if raw_decoded and raw_decoded.strip():
                        total_triples += ingest_python_ast_to_neo4j(raw_decoded, source_doc=filename)

                        code_chunks = chunk_python_ast(raw_decoded)
                        points = []
                        for idx, chunk_str in enumerate(code_chunks):
                            chunk_id = str(uuid.uuid4())
                            emb_res = ollama_client.embeddings(model="nomic-embed-text", prompt=chunk_str)
                            points.append(
                                PointStruct(
                                    id=chunk_id,
                                    vector=emb_res["embedding"],
                                    payload={"text": chunk_str, "source": filename, "chunk_index": idx, "is_code": True}
                                )
                            )

                        if points:
                            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                            total_chunks += len(code_chunks)
                            processed_files.append(filename)

                        continue
                except Exception as e:
                    print(f"[Python AST Ingest Error] Falling back to text mode for {filename}: {e}")

            # --- 4. DEDICATED YAML SPEC ROUTE (.yaml, .yml) ---
            if ext in [".yaml", ".yml"]:
                try:
                    raw_decoded = decode_bytes(file_bytes)
                    yaml_payload = yaml.safe_load(raw_decoded)

                    if yaml_payload:
                        total_triples += ingest_json_to_neo4j(yaml_payload, source_doc=filename)

                        yaml_chunks = chunk_yaml_by_sections(yaml_payload)
                        points = []
                        for idx, chunk_str in enumerate(yaml_chunks):
                            chunk_id = str(uuid.uuid4())
                            emb_res = ollama_client.embeddings(model="nomic-embed-text", prompt=chunk_str)
                            points.append(
                                PointStruct(
                                    id=chunk_id,
                                    vector=emb_res["embedding"],
                                    payload={"text": chunk_str, "source": filename, "chunk_index": idx, "is_yaml": True}
                                )
                            )

                        if points:
                            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                            total_chunks += len(yaml_chunks)
                            processed_files.append(filename)

                        continue
                except Exception as e:
                    print(f"[YAML Ingest Error] Falling back to text mode for {filename}: {e}")

            # --- 5. DEDICATED EMAIL ROUTE (.eml) ---
            if ext == ".eml":
                try:
                    email_data = parse_eml_file(file_bytes)
                    total_triples += ingest_email_to_neo4j(email_data, source_doc=filename)

                    email_chunks = chunk_email_content(email_data)
                    points = []
                    for idx, chunk_str in enumerate(email_chunks):
                        chunk_id = str(uuid.uuid4())
                        emb_res = ollama_client.embeddings(model="nomic-embed-text", prompt=chunk_str)
                        points.append(
                            PointStruct(
                                id=chunk_id,
                                vector=emb_res["embedding"],
                                payload={"text": chunk_str, "source": filename, "chunk_index": idx, "is_email": True}
                            )
                        )

                    if points:
                        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                        total_chunks += len(email_chunks)
                        processed_files.append(filename)

                    continue
                except Exception as e:
                    print(f"[Email Ingest Error] Falling back to text mode for {filename}: {e}")

            # --- 6. DEDICATED LOG FILE ROUTE (.log) ---
            if ext == ".log":
                try:
                    raw_decoded = decode_bytes(file_bytes)
                    if raw_decoded and raw_decoded.strip():
                        log_entries = parse_log_entries(raw_decoded)
                        total_triples += ingest_logs_to_neo4j(log_entries, source_doc=filename)

                        log_chunks = chunk_log_entries(log_entries, max_entries_per_chunk=10)
                        points = []
                        for idx, chunk_str in enumerate(log_chunks):
                            chunk_id = str(uuid.uuid4())
                            emb_res = ollama_client.embeddings(model="nomic-embed-text", prompt=chunk_str)
                            points.append(
                                PointStruct(
                                    id=chunk_id,
                                    vector=emb_res["embedding"],
                                    payload={"text": chunk_str, "source": filename, "chunk_index": idx, "is_log": True}
                                )
                            )

                        if points:
                            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                            total_chunks += len(log_chunks)
                            processed_files.append(filename)

                        continue
                except Exception as e:
                    print(f"[Log Ingest Error] Falling back to text mode for {filename}: {e}")

            # --- 7. STANDARD UNSTRUCTURED TEXT / PROSE ROUTE ---
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

        # Check if text payload is direct raw JSON
        if raw_text.startswith("{") or raw_text.startswith("["):
            try:
                json_payload = json.loads(raw_text)
                total_triples += ingest_json_to_neo4j(json_payload, source_doc="raw_json_text")
                json_chunks = chunk_json_object_level(json_payload)

                points = []
                for idx, chunk_str in enumerate(json_chunks):
                    chunk_id = str(uuid.uuid4())
                    emb_res = ollama_client.embeddings(model="nomic-embed-text", prompt=chunk_str)
                    points.append(
                        PointStruct(
                            id=chunk_id,
                            vector=emb_res["embedding"],
                            payload={"text": chunk_str, "source": "raw_json_text", "chunk_index": idx, "is_json": True}
                        )
                    )

                qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                return {
                    "status": "success",
                    "filename": "raw_json_text",
                    "chunks_processed": len(json_chunks),
                    "vector_status": f"Indexed {len(json_chunks)} chunks in Qdrant",
                    "graph_status": f"Stored {total_triples} relationship triples in Neo4j"
                }
            except Exception:
                pass

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