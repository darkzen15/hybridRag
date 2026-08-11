HYBRID RAG STACK: COMPLETE SYSTEM SETUP & INSTRUCTION GUIDE
================================================================================

1. SYSTEM ARCHITECTURE & SPECIALIZED INGESTION PIPELINES
--------------------------------------------------------------------------------
The API gateway (main.py) uses 7 specialized ingestion pipelines to preserve 
context, syntax, and relational topology before storing vectors in Qdrant and 
triples in Neo4j.

Data Type          Formats               Neo4j Strategy                         Qdrant Strategy
------------------------------------------------------------------------------------------------------------------------
Structured JSON    .json, .jsonl         Recursive JSON tree mapping            Object-level boundary chunking
Tabular Data       .csv, .tsv, .xlsx     Row entity & category mapping          Multi-row blocks + header repetition
Source Code        .py                   AST parsing (classes, funcs, imports)  Class/Function AST code blocks
Config / Specs     .yaml, .yml           Tree mapping via dict parser           Section-level YAML dumps
Emails             .eml                  Message topology (from, to, replies)   Header summary + body chunks
System Logs        .log                  Log events, levels, exception types    Multi-line log entry blocks
Prose / Docs       .pdf, .docx, .txt     spaCy NER + Ollama LLM extraction      Sliding window (1000 char / 100 overlap)


2. PROJECT FILES
--------------------------------------------------------------------------------

=== File 1: requirements.txt ===
fastapi>=0.110.0
uvicorn[standard]>=0.28.0
pydantic>=2.6.0
python-multipart>=0.0.9
qdrant-client>=1.8.0
neo4j>=5.18.0
ollama>=0.1.7
pypdf>=4.1.0
python-docx>=1.1.0
spacy>=3.7.0
pandas>=2.2.0
openpyxl>=3.1.2
PyYAML>=6.0.1

=== File 2: Dockerfile.api ===
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English NLP model
RUN python -m spacy download en_core_web_sm

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

=== File 3: docker-compose.yml ===
version: "3.8"

services:
  neo4j:
    image: neo4j:5.18.0
    container_name: neo4j-rag
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password123
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
      - NEO4J_dbms_security_procedures_allowlist=apoc.*
    volumes:
      - neo4j_data:/data

  qdrant:
    image: qdrant/qdrant:v1.8.2
    container_name: qdrant-rag
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  hybrid-rag-api:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: hybrid-rag-api
    ports:
      - "8000:8000"
    environment:
      - QDRANT_HOST=qdrant-rag
      - NEO4J_URI=bolt://neo4j-rag:7687
      - OLLAMA_HOST=http://host.docker.internal:11434
    depends_on:
      - neo4j
      - qdrant
    extra_hosts:
      - "host.docker.internal:host-gateway"

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    ports:
      - "3000:8080"
    environment:
      - OPENAI_API_BASE_URL=http://hybrid-rag-api:8000/v1
      - OPENAI_API_KEY=sk-dummy-key
      - ENABLE_OLLAMA_INTEGRATION=False
    volumes:
      - open_webui_data:/app/backend/data
    depends_on:
      - hybrid-rag-api

volumes:
  neo4j_data:
  qdrant_data:
  open_webui_data:


3. BUILD & DEPLOYMENT INSTRUCTIONS
--------------------------------------------------------------------------------
Standard Launch:
  docker compose up -d --build

Fix Docker BuildKit Credential Helper Error (if triggered during build):
  Windows (PowerShell):
    Set-Content -Path "$env:USERPROFILE\.docker\config.json" -Value '{"auths":{}}'
    docker pull python:3.11-slim
    docker compose up -d --build

  Linux / macOS:
    echo '{"auths":{}}' > ~/.docker/config.json
    docker pull python:3.11-slim
    docker compose up -d --build

Complete Fresh Database Wipe:
  docker compose down -v
  docker compose up -d --build


4. OPEN WEBUI DETAILED CONFIGURATION FOR HYBRID RAG
--------------------------------------------------------------------------------
To route all queries through the custom Graph + Vector RAG gateway without 
bypassing the custom ingestion pipeline, configure Open WebUI as follows:

A. Environment Setup (Pre-configured in docker-compose.yml):
   1. OPENAI_API_BASE_URL: Set to "http://hybrid-rag-api:8000/v1"
      This directs Open WebUI chat completions to the FastAPI middleware.
   2. OPENAI_API_KEY: Set to "sk-dummy-key" (Required by Open WebUI client).
   3. ENABLE_OLLAMA_INTEGRATION: Set to "False"
      CRITICAL: Disabling native Ollama integration prevents Open WebUI from 
      connecting directly to Ollama and bypassing the Neo4j/Qdrant retrieval loop.

B. First-Time UI Setup:
   1. Navigate to http://localhost:3000 in your browser.
   2. Create your initial admin account.
   3. Open "Admin Panel" -> "Settings" -> "Connections".
   4. Under "OpenAI API", verify URL is set to: http://hybrid-rag-api:8000/v1
   5. Click the refresh/verify icon next to the URL input field. You should see 
      models populated from the gateway (e.g., llama3.2).

C. Disable Open WebUI Native RAG (To Avoid Bypassing Custom Pipelines):
   1. Open "Admin Panel" -> "Settings" -> "Documents".
   2. Set "RAG Embedding Engine" to "None" or disable auto-indexing.
   3. Disable native Web Search inside Open WebUI settings.
   4. Reason: Open WebUI's native upload button uses an internal ChromaDB instance,
      which lacks support for Neo4j triples and specialized parsers (AST, EML, LOG).

D. Operating Model & Document Upload Rule:
   1. In the chat interface, select "llama3.2" (or your specified Ollama model) 
      from the top model drop-down menu.
   2. ALL document uploads (PDF, JSON, CSV, PY, YAML, EML, LOG) MUST be sent 
      directly to the FastAPI endpoint (http://localhost:8000/ingest) or via your 
      Ingest GUI—NOT attached via Open WebUI's chat paperclip icon.
   3. Once uploaded to /ingest, ask your question directly in the Open WebUI 
      chatbox. The middleware automatically performs hybrid retrieval (Neo4j + 
      Qdrant) and injects the context into the LLM response stream.


5. API TESTING & VERIFICATION COMMANDS
--------------------------------------------------------------------------------
Ingest Direct Text Payload:
  curl -X POST "http://localhost:8000/ingest" \
    -F "text=FastAPI communicates with Qdrant for vector storage and Neo4j for graph storage."

Ingest Structured Files:
  curl -X POST "http://localhost:8000/ingest" \
    -F "files=@data.json" \
    -F "files=@metrics.csv" \
    -F "files=@app.py"

List Available Models:
  curl -X GET "http://localhost:8000/v1/models"

Test Streaming / Non-Streaming Completion:
  curl -X POST "http://localhost:8000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "llama3.2",
      "messages": [{"role": "user", "content": "What database stores vectors?"}],
      "stream": false
    }'

Inspect Knowledge Graph in Neo4j Browser (http://localhost:7474):
  MATCH (a:Entity)-[r]->(b:Entity)
  RETURN a.name AS Subject, labels(a) AS SubjectType, type(r) AS Predicate, b.name AS Object, labels(b) AS ObjectType;

Clear Neo4j Graph Database:
  MATCH (n) DETACH DELETE n;