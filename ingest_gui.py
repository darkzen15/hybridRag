import datetime
import requests
import streamlit as st

st.set_page_config(
    page_title="Hybrid RAG Ingester",
    page_icon="📥",
    layout="wide"
)

# Initialize Session State Audit History
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
api_url = st.sidebar.text_input("Ingestion Endpoint URL", value="http://hybrid-rag-api:8000/ingest")

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
tab1, tab2, tab3 = st.tabs(["📁 Bulk File Upload", "📝 Raw Text Input", "📜 Session History"])

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
            current_progress = idx / total_files
            progress_bar.progress(current_progress, text=f"Processing ({idx + 1}/{total_files}): {file.name}")
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            try:
                files = {"file": (file.name, file.getvalue(), file.type)}
                response = requests.post(api_url, files=files, timeout=300)

                if response.status_code == 200:
                    data = response.json()
                    successful_count += 1
                    st.session_state.ingestion_history.insert(0, {
                        "Timestamp": timestamp,
                        "Source": file.name,
                        "Type": file.name.split(".")[-1].upper(),
                        "Status": "Success ✅",
                        "Chunks": data.get("chunks_processed", 0),
                        "Vector Status": data.get("vector_status", "Indexed"),
                        "Graph Status": data.get("graph_status", "Stored")
                    })
                else:
                    failed_count += 1
                    st.session_state.ingestion_history.insert(0, {
                        "Timestamp": timestamp,
                        "Source": file.name,
                        "Type": file.name.split(".")[-1].upper(),
                        "Status": f"Failed ({response.status_code}) ❌",
                        "Chunks": 0,
                        "Vector Status": response.text[:50],
                        "Graph Status": "Failed"
                    })
            except Exception as e:
                failed_count += 1
                st.session_state.ingestion_history.insert(0, {
                    "Timestamp": timestamp,
                    "Source": file.name,
                    "Type": file.name.split(".")[-1].upper(),
                    "Status": "Exception ❌",
                    "Chunks": 0,
                    "Vector Status": str(e)[:50],
                    "Graph Status": "Error"
                })

        progress_bar.progress(1.0, text="Bulk processing complete!")
        status_container.success(f"Processed {total_files} document(s): **{successful_count} succeeded**, **{failed_count} failed**.")

# --- TAB 2: RAW TEXT INPUT ---
with tab2:
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
                        "Chunks": res_data.get("chunks_processed", 0),
                        "Vector Status": res_data.get("vector_status", "Indexed"),
                        "Graph Status": res_data.get("graph_status", "Stored")
                    })
                else:
                    st.error(f"Failed (Status {response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

# --- TAB 3: AUDIT LOG ---
with tab3:
    st.subheader("📜 Ingestion Audit Log")
    if st.session_state.ingestion_history:
        st.dataframe(
            st.session_state.ingestion_history,
            use_container_width=True,
            column_config={
                "Timestamp": st.column_config.TextColumn("Time", width="small"),
                "Source": st.column_config.TextColumn("Source Name", width="medium"),
                "Type": st.column_config.TextColumn("Format", width="small"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Chunks": st.column_config.NumberColumn("Chunks Processed"),
                "Vector Status": st.column_config.TextColumn("Qdrant Vector Response"),
                "Graph Status": st.column_config.TextColumn("Neo4j Graph Response"),
            }
        )
    else:
        st.info("No documents or text payloads have been ingested during this session yet.")