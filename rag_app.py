import ollama
import streamlit as st
import os
import pdfplumber
import sqlite3
from sentence_transformers import SentenceTransformer
from langchain.llms import Ollama
import numpy as np
import faiss

models_info = ollama.list()

def ollama_models(models_info):
    embedding_models = [model["name"] for model in models_info["models"] if 'embed' in model["name"]]
    llm_models = [model["name"] for model in models_info["models"] if 'embed' not in model["name"]]
    print(embedding_models, llm_models)
    return embedding_models, llm_models

# Initialize SQLite database
def init_db(db_name):
    if not os.path.exists(db_name):
        conn = sqlite3.connect(db_name)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS embeddings
                     (id INTEGER PRIMARY KEY, segment TEXT, embedding BLOB, model TEXT, source TEXT)''')
        conn.commit()
        conn.close()

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf_file:
        for page in pdf_file.pages:
            text += page.extract_text() + "\n"
    return text.strip()

# Step 2: Split the Text into Segments with Context
def split_text(text, chunk_size=10):
    segments = []
    sentences = text.split('. ')
    for i in range(0, len(sentences), chunk_size):
        segment = ' '.join(sentences[i:i + chunk_size])
        if len(segment.split()) > 1024:
            segments.append(segment[:1024])
        else:
            segments.append(segment)
    return segments

# Step 3: Create Embeddings
def create_embeddings(segments, model_name):
    model = SentenceTransformer(model_name)
    batch_size = 16
    embeddings = []
    for i in range(0, len(segments), batch_size):
        batch_embeddings = model.encode(segments[i:i + batch_size])
        embeddings.extend(batch_embeddings)
    val = np.array(embeddings)
    print(val.shape)
    return val

# Step 4: Store Embeddings in Database
def store_embeddings_in_db(segments, embeddings, model_name, sources, db_path):
    conn = sqlite3.connect(db_path)  # Use the provided database name
    c = conn.cursor()
    for segment, embedding, source in zip(segments, embeddings, sources):
        c.execute("INSERT INTO embeddings (segment, embedding, model, source) VALUES (?, ?, ?, ?)",
                  (segment, embedding.tobytes(), model_name, source))
    conn.commit()
    conn.close()

# Step 5: Retrieve Embeddings from Database
def retrieve_embeddings_from_db(model_name, db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT segment, embedding, source FROM embeddings WHERE model=?", (model_name,))
    rows = c.fetchall()
    segments = [row[0] for row in rows]
    embeddings = [np.frombuffer(row[1], dtype='float32') for row in rows]
    sources = [row[2] for row in rows]
    conn.close()
    return segments, np.array(embeddings), sources

# Step 6: Index embeddings using FAISS
def index_embeddings(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Step 7: Retrieval Mechanism
def retrieve_relevant_segments(embeddings, query, segments, sources, model_name, index, top_k=3):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query]).astype('float32')
    _, indices = index.search(query_embedding, top_k)
    print(query_embedding.shape)
    return [(segments[idx], embeddings[idx], sources[idx]) for idx in indices[0]]

# Step 8: Generate Response
def generate_response(prompts, llm_model):
    # Ensure that prompts is a list of strings
    if isinstance(prompts, str):
        prompts = [prompts]

    # Create an instance of the Ollama class
    ollama_instance = Ollama(model=llm_model)

    # Call the generate method on the instance
    response = ollama_instance.generate(prompts=prompts)

    # Extracting the text from the response assuming a nested structure with 'generations'
    # Adjust this based on the actual structure you have seen in your response
    generated_text = ""
    if response and response.generations:
        # Assuming generations is a list of GenerationChunk objects
        generated_text = response.generations[0][0].text if response.generations[0] else "No response generated."

    return generated_text

# Streamlit Interface
def main():
    st.title("RAG Application with Streamlit")

    embedding_models = ["multi-qa-mpnet-base-dot-v1", "all-MiniLM-L6-v2", "hkunlp/instructor-xl"]
    _, llm_models = ollama_models(models_info)

    # Create folders for PDFs and database if they don't exist
    os.makedirs("uploaded_pdfs", exist_ok=True)
    os.makedirs("database", exist_ok=True)

    # Create a space for uploading PDFs or selecting existing embeddings
    st.header("Select or Create Embeddings")

    # Option to upload PDFs for new embeddings
    st.subheader("Upload PDF Files for New Embeddings")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    embedding_model_selected = st.selectbox("Select Embedding Model for Processing", embedding_models)

    db_name = st.text_input("Enter a name for the new database:", "database/embeddings.db")  # New input for DB name

    if st.button("Process New PDFs") and uploaded_files and embedding_model_selected and db_name:
        init_db(db_name)  # Initialize the new database

        all_text_segments = []
        all_sources = []

        for uploaded_file in uploaded_files:
            source = uploaded_file.name
            with open(os.path.join("uploaded_pdfs", source), "wb") as f:
                f.write(uploaded_file.getbuffer())

            pdf_path = os.path.join("uploaded_pdfs", source)
            document_text = extract_text_from_pdf(pdf_path)
            text_segments = split_text(document_text)
            all_text_segments.extend(text_segments)
            all_sources.extend([source] * len(text_segments))

        embeddings = create_embeddings(all_text_segments, embedding_model_selected)
        store_embeddings_in_db(all_text_segments, embeddings, embedding_model_selected, all_sources, db_name)
        st.success(f"PDFs processed and embeddings stored successfully in {db_name}. Total segments: {len(all_text_segments)}")

    # Option to select existing embeddings
    st.subheader("Select Existing Embeddings Database")
    db_files = [f for f in os.listdir("database") if f.endswith('.db')]
    selected_db = st.selectbox("Select a database:", [""] + db_files)

    if selected_db:
        db_path = os.path.join("database", selected_db)
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT DISTINCT model FROM embeddings")
        models = [row[0] for row in c.fetchall()]
        conn.close()

        model_selected = st.selectbox("Select an embedding model:", [""] + models)

        if model_selected:
            segments, embeddings, sources = retrieve_embeddings_from_db(model_selected, db_path)
            if segments:
                st.success(f"Loaded embeddings for model: {model_selected}. Total segments: {len(segments)}")
                index = index_embeddings(embeddings)
                st.session_state["index"] = index
                st.session_state["segments"] = segments
                st.session_state["embeddings"] = embeddings
                st.session_state["sources"] = sources
            else:
                st.warning("No segments found for the selected model.")

    # Query Section
    st.header("Query from Embeddings")
    if "segments" in st.session_state:
        query = st.text_input("Enter your query:")
        llm_model_selected = st.selectbox("Select LLM Model for Query", llm_models)

        if st.button("Query") and query:
            if "index" in st.session_state and "segments" in st.session_state and "embeddings" in st.session_state:
                index = st.session_state["index"]
                segments = st.session_state["segments"]
                embeddings = st.session_state["embeddings"]
                sources = st.session_state["sources"]

                st.write(f"Embedding Model: {model_selected}, LLM Model: {llm_model_selected}")
                st.write(f"**Query:** <span style='color:red'>{query}</span>", unsafe_allow_html=True)

                results = retrieve_relevant_segments(embeddings, query, segments, sources, model_selected, index)
                print("The Results are: ", results)

                for i, result in enumerate(results):
                    best_prompt = f"Based on the following retrieved segment, please provide the most relevant and accurate response to the query: {query}\n\nSegment: {result[0]}"
                    print("THE MODEL IS : ", llm_model_selected)
                    print("THE PROMPT IS :", best_prompt)
                    generated_text = generate_response(best_prompt, llm_model_selected)
                    st.markdown(
                        f"<span style='color:blue'>**Retrieved Segment:** {result[0]}</span><br><br>"
                        f"<span style='color:green; font-style:italic;'>**Generated Response:** {generated_text}</span><br>"
                        f"**Source:** [Link]({result[2]})<br><hr>",
                        unsafe_allow_html=True
                    )
                    if i == 0:
                        st.markdown("<hr>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
