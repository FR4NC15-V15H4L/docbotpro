import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# ====== CONFIG ======
genai.configure(api_key=st.secrets["gemini"]["api_key"])
CHUNK_SIZE = 500
TOP_K = 3

# ====== GEMINI SETUP ======
genai.configure(api_key=GEMINI_API_KEY)
model_embedder = SentenceTransformer('all-MiniLM-L6-v2')
gen_model = genai.GenerativeModel("models/gemini-1.5-flash")

# ====== SIDEBAR ======
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Google_Gemini_logo.svg/512px-Google_Gemini_logo.svg.png", width=150)
st.sidebar.title("📚 RAG Chatbot")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or TXT", type=["pdf", "txt"])

# ====== MAIN PANEL ======
st.title("💬 Ask Your Document (with RAG + Gemini)")

if uploaded_file:
    # Extract text
    file_text = ""
    if uploaded_file.name.endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        file_text = "\n".join(page.get_text() for page in doc)
    else:
        file_text = uploaded_file.read().decode("utf-8")
    
    st.sidebar.success(f"✅ {uploaded_file.name} uploaded")

    # Preview doc (collapsible)
    with st.expander("📄 Document Preview", expanded=False):
        st.text_area("Contents", file_text[:3000] + ("..." if len(file_text) > 3000 else ""), height=200)

    # ====== CHUNKING & EMBEDDINGS ======
    def chunk_text(text, chunk_size=CHUNK_SIZE):
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    with st.spinner("🔍 Indexing your document..."):
        chunks = chunk_text(file_text)
        embeddings = model_embedder.encode(chunks)
        index = faiss.IndexFlatL2(embeddings[0].shape[0])
        index.add(np.array(embeddings))

    st.success("✅ Document is ready. Ask away!")

    # ====== SESSION STATE FOR CHAT HISTORY ======
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("💡 Ask a question about the document", placeholder="e.g. What is the summary of the paper?")
    if query:
        query_embedding = model_embedder.encode([query])
        _, I = index.search(np.array(query_embedding), TOP_K)
        relevant_chunks = [chunks[i] for i in I[0]]
        context = "\n\n".join(relevant_chunks)

        prompt = f"""Use the context below to answer the question as clearly as possible.

        Context:
        {context}

        Question:
        {query}
        """

        with st.spinner("🤖 Gemini is thinking..."):
            response = gen_model.generate_content(prompt)
            answer = response.text

        st.session_state.chat_history.append((query, answer))

    # ====== DISPLAY CHAT HISTORY ======
    st.markdown("### 🧠 Chat History")
    for q, a in st.session_state.chat_history[::-1]:
        with st.container():
            st.markdown(f"**🟦 You:** {q}")
            st.markdown(f"**🟨 Gemini:** {a}")
            st.markdown("---")

else:
    st.info("👈 Upload a document from the sidebar to get started!")
