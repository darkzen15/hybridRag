import datetime
import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Hybrid RAG Ingester",
    page_icon="📥",
    layout="wide"
)

if "ingestion_history" not in st.session_state:
    st.session_state.ingestion_history = []

st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.5rem; }
    .sub-title { color: #6c757d; margin-bottom: 1.5rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📥 Hybrid RAG Ingestion Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Parse documents and load vector embeddings into <b>Qdrant</b> alongside knowledge graph relationships into <b>Neo4j</b>.</div>', unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("⚙️ Gateway Configuration")
default_api = os.getenv("API_URL", "http://hybrid-rag-api:8000/ingest")
api_url = st.sidebar.text_input("Ingestion Endpoint URL", value=default_api)

if st.sidebar.button("🔌 Test API Connection"):
    try:
        health_check_url = api_url.replace("/ingest", "/v1/models")
        res = requests.get(health_check_url, timeout=3)
        if res.status_code == 200:
            st.sidebar.success("API Gateway Online 🟢")
        else:
            st.sidebar.warning(f"API returned status code {res.status_code}")
    except Exception as e:
        st.sidebar.error(f"Connection failed: {e}")

st.sidebar.divider()
st.sidebar.header("📊 Session Metrics")
st.sidebar.metric("Total Ingested Items", len(st.session_state.ingestion_history))

if st.sidebar.button("🗑️ Clear Ingestion History", use_container_width=True):
    st.session_state.ingestion_history = []
    st.sidebar.info("History cleared!")
    st.rerun()

# Layout Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📁 Bulk File Upload", "🖥️ Local Directory Scan", "📝 Raw Text", "📜 Audit Log"])

# --- TAB 1: BULK FILE UPLOAD ---
with tab1:
    uploaded_files = st.file_uploader(
        "Drag and drop documents here",
        type=["pdf", "docx", "doc", "txt", "md"],
        accept_multiple_files=True
    )

    if st.button("Start Bulk Ingestion", type="primary", disabled=not uploaded_files):
        total_files = len(uploaded_files)
        progress_bar = st.progress(0, text="Initializing upload queue...")
        status_container = st.empty()
        
        successful_count = 0
        failed_count = 0

        for idx, file in enumerate(uploaded_files):
            progress_bar.progress(idx / total_files, text=f"Processing ({idx + 1}/{total_files}): {file.name}")
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            try:
                files = [("files", (file.name, file.getvalue(), file.type))]
                response = requests.post(api_url, files=files, timeout=300)

                if response.status_code == 200:
                    data = response.json()
                    successful_count += 1
                    st.session_state.ingestion_history.insert(0, {
                        "Timestamp": timestamp,
                        "Source": file.name,
                        "Type": file.name.split(".")[-1].upper(),
                        "Status": "Success ✅",
                        "Chunks": data.get("chunks_processed", 0)
                    })
                else:
                    failed_count += 1
                    st.session_state.ingestion_history.insert(0, {
                        "Timestamp": timestamp,
                        "Source": file.name,
                        "Type": file.name.split(".")[-1].upper(),
                        "Status": f"Failed ({response.status_code}) ❌",
                        "Chunks": 0
                    })
            except Exception as e:
                failed_count += 1
                st.session_state.ingestion_history.insert(0, {
                    "Timestamp": timestamp,
                    "Source": file.name,
                    "Type": file.name.split(".")[-1].upper(),
                    "Status": "Exception ❌",
                    "Chunks": 0
                })

        progress_bar.progress(1.0, text="Bulk processing complete!")
        status_container.success(f"Processed {total_files} document(s): **{successful_count} succeeded**, **{failed_count} failed**.")

# --- TAB 2: LOCAL DIRECTORY SCANNER ---
with tab2:
    st.markdown("##### Scan and ingest a local folder path on the system host")
    folder_path = st.text_input("Absolute Directory Path", placeholder="/path/to/my_documents or C:\\Documents")

    if st.button("Scan & Ingest Folder", type="primary", disabled=not folder_path.strip()):
        if not os.path.exists(folder_path):
            st.error(f"Directory path not found: `{folder_path}`")
        elif not os.path.isdir(folder_path):
            st.error(f"Provided path is a file, not a directory: `{folder_path}`")
        else:
            supported_exts = (".pdf", ".docx", ".doc", ".txt", ".md")
            found_files = []

            for root, _, files in os.walk(folder_path):
                for f in files:
                    if f.lower().endswith(supported_exts):
                        found_files.append(os.path.join(root, f))

            if not found_files:
                st.warning("No supported files (.pdf, .docx, .txt, .md) found in this directory.")
            else:
                st.info(f"Found **{len(found_files)}** document(s) in directory. Processing...")
                prog = st.progress(0, text="Starting folder scan...")

                for idx, fpath in enumerate(found_files):
                    fname = os.path.basename(fpath)
                    prog.progress(idx / len(found_files), text=f"Ingesting ({idx+1}/{len(found_files)}): {fname}")
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

                    try:
                        with open(fpath, "rb") as f_obj:
                            files = [("files", (fname, f_obj.read()))]
                            res = requests.post(api_url, files=files, timeout=300)

                            if res.status_code == 200:
                                data = res.json()
                                st.session_state.ingestion_history.insert(0, {
                                    "Timestamp": timestamp,
                                    "Source": fname,
                                    "Type": fname.split(".")[-1].upper(),
                                    "Status": "Success ✅",
                                    "Chunks": data.get("chunks_processed", 0)
                                })
                            else:
                                st.session_state.ingestion_history.insert(0, {
                                    "Timestamp": timestamp,
                                    "Source": fname,
                                    "Type": fname.split(".")[-1].upper(),
                                    "Status": f"Failed ({res.status_code}) ❌",
                                    "Chunks": 0
                                })
                    except Exception as e:
                        st.session_state.ingestion_history.insert(0, {
                            "Timestamp": timestamp,
                            "Source": fname,
                            "Type": fname.split(".")[-1].upper(),
                            "Status": f"Error: {e}",
                            "Chunks": 0
                        })

                prog.progress(1.0, text="Folder scan complete!")
                st.success(f"Successfully processed folder: `{folder_path}`")

# --- TAB 3: RAW TEXT INPUT ---
with tab3:
    raw_text = st.text_area("Paste raw text or Markdown content", height=250, placeholder="Paste notes, transcripts, or facts...")

    if st.button("Ingest Text Payload", type="primary", disabled=not raw_text.strip()):
        with st.spinner("Extracting triples and generating vector embeddings..."):
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            try:
                response = requests.post(api_url, data={"text": raw_text}, timeout=300)

                if response.status_code == 200:
                    res_data = response.json()
                    st.success("Successfully ingested text payload!")
                    st.session_state.ingestion_history.insert(0, {
                        "Timestamp": timestamp,
                        "Source": "Raw Text String",
                        "Type": "TEXT",
                        "Status": "Success ✅",
                        "Chunks": res_data.get("chunks_processed", 0)
                    })
                else:
                    st.error(f"Failed (Status {response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

# --- TAB 4: AUDIT LOG ---
with tab4:
    st.subheader("📜 Ingestion Audit Log")
    if st.session_state.ingestion_history:
        st.dataframe(st.session_state.ingestion_history, use_container_width=True)
    else:
        st.info("No documents or text payloads have been ingested during this session yet.")