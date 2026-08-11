import sys
import requests

API_URL = "http://localhost:8000/ingest"

def ingest_file_or_text(target: str):
    try:
        if target.endswith((".pdf", ".docx", ".doc", ".txt", ".md")):
            with open(target, "rb") as f:
                files = {"file": (target, f)}
                res = requests.post(API_URL, files=files, timeout=300)
        else:
            res = requests.post(API_URL, data={"text": target}, timeout=300)

        if res.status_code == 200:
            print("Ingestion Successful!")
            print(res.json())
        else:
            print(f"Failed with status code {res.status_code}: {res.text}")
    except Exception as e:
        print(f"Error connecting to ingestion server: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ingest_file_or_text(sys.argv[1])
    else:
        ingest_file_or_text("Neo4j integrates with Qdrant and Ollama inside a hybrid RAG architecture.")