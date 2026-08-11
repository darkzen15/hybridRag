import os
import sys
import requests

API_URL = "http://localhost:8000/ingest"
SUPPORTED_EXTENSIONS = (
    ".pdf", ".docx", ".doc", ".txt", ".md", ".json", ".jsonl",
    ".csv", ".tsv", ".xml", ".yaml", ".yml", ".log", ".py",
    ".js", ".ts", ".html", ".css", ".sql", ".sh", ".env", ".ini",
    ".conf", ".c", ".cpp", ".h", ".go", ".rs", ".java"
)


def ingest_file(file_path: str):
    """Ingest a single file."""
    filename = os.path.basename(file_path)
    try:
        with open(file_path, "rb") as f:
            files = [("files", (filename, f))]
            res = requests.post(API_URL, files=files, timeout=300)
            if res.status_code == 200:
                print(f"  ✅ Ingested: {filename}")
            else:
                print(f"  ❌ Failed ({res.status_code}): {filename} - {res.text[:100]}")
    except Exception as e:
        print(f"  ❌ Connection error processing {filename}: {e}")


def ingest_folder(folder_path: str):
    """Recursively scan a directory and ingest all valid files."""
    print(f"\n📂 Scanning directory: {folder_path}")
    matched_files = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTENSIONS):
                matched_files.append(os.path.join(root, file))

    if not matched_files:
        print("⚠️ No supported text/code/document files found in this folder.")
        return

    print(f"Found {len(matched_files)} file(s). Starting batch ingestion...\n")
    for idx, path in enumerate(matched_files, start=1):
        print(f"[{idx}/{len(matched_files)}] Ingesting...")
        ingest_file(path)

    print("\n🎉 Folder ingestion complete!")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python ingest.py <file_path_or_folder_path>")
        print("  python ingest.py \"raw text to ingest directly\"")
        sys.exit(1)

    target = sys.argv[1]

    if os.path.isdir(target):
        ingest_folder(target)
    elif os.path.isfile(target):
        print(f"Ingesting single file: {target}")
        ingest_file(target)
    else:
        print("Ingesting raw text payload...")
        try:
            res = requests.post(API_URL, data={"text": target}, timeout=300)
            print(res.json())
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()