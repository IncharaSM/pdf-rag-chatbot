import os
import numpy as np
from numpy.linalg import norm
from dotenv import load_dotenv
load_dotenv()
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, render_template
from groq import Groq
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
client = Groq(api_key=os.environ.get("your_api_key"))

# Load embedding model once at startup
print("Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model ready.")

pdf_text_store = {}  # { filename: { "chunks": [...], "embeddings": np.array } }

def extract_text_from_pdf(file):
    data = file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=3000):
    words = text.split()
    chunks, current, count = [], [], 0
    for word in words:
        current.append(word)
        count += len(word) + 1
        if count >= chunk_size:
            chunks.append(" ".join(current))
            current, count = [], 0
    if current:
        chunks.append(" ".join(current))
    return chunks

def find_relevant_chunks_vector(question, top_n=4):
    """Find most relevant chunks across all PDFs using cosine similarity."""
    query_vec = embed_model.encode([question], convert_to_numpy=True)[0]
    results = []
    for filename, data in pdf_text_store.items():
        for chunk, emb in zip(data["chunks"], data["embeddings"]):
            # Cosine similarity
            score = np.dot(query_vec, emb) / (norm(query_vec) * norm(emb) + 1e-10)
            results.append((score, filename, chunk))
    results.sort(reverse=True)
    return results[:top_n]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file received. Make sure you selected a PDF."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400

    try:
        text = extract_text_from_pdf(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read PDF: {str(e)}"}), 500

    if not text.strip():
        return jsonify({"error": "PDF appears to be empty or scanned (no text found)."}), 400

    chunks = chunk_text(text)

    # Generate vector embeddings for all chunks
    print(f"Embedding {len(chunks)} chunks for '{file.filename}'...")
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    print(f"Done embedding '{file.filename}'.")

    pdf_text_store[file.filename] = {
        "chunks": chunks,
        "embeddings": embeddings
    }

    return jsonify({
        "message": f"'{file.filename}' loaded! {len(pdf_text_store)} PDF(s) total."
    })

@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "pdf_count": len(pdf_text_store),
        "pdf_names": list(pdf_text_store.keys())
    })

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400
    if not pdf_text_store:
        return jsonify({"error": "Please upload a PDF first."}), 400

    # Vector search across all PDFs
    top_chunks = find_relevant_chunks_vector(question, top_n=4)
    all_chunks = [f"[From: {fname}]\n{chunk}" for _, fname, chunk in top_chunks]
    context = "\n\n---\n\n".join(all_chunks)

    prompt = f"""You have been given content from {len(pdf_text_store)} PDF(s), each labeled with its filename.
Answer the question based ONLY on the content below.
If asked for separate summaries, summarize each PDF separately using its filename as a header.
If the answer is not found, say "I couldn't find that in the PDFs."

PDF Content:
{context}

Question: {question}
Answer:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2
        )
        return jsonify({"answer": response.choices[0].message.content.strip()})
    except Exception as e:
        return jsonify({"error": f"Groq API error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))