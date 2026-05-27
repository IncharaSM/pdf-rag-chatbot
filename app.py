import os
from dotenv import load_dotenv
load_dotenv()
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, render_template
from groq import Groq
#
app = Flask(__name__)
client = Groq(api_key= os.environ.get("your_api_key"))

pdf_text_store = {}

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

def find_relevant_chunks(chunks, question, top_n=3):
    question_words = set(question.lower().split())
    scored = sorted(chunks, key=lambda c: len(set(c.lower().split()) & question_words), reverse=True)
    return scored[:top_n]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    print("Files received:", request.files)
    print("Form data:", request.form)

    if "file" not in request.files:
        return jsonify({"error": "No file received. Make sure you selected a PDF."}), 400

    file = request.files["file"]
    print("Filename:", file.filename)

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

    pdf_text_store["current"] = chunk_text(text)
    return jsonify({
        "message": f"PDF loaded! {len(text)} characters across {len(pdf_text_store['current'])} chunks."
    })

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400
    if "current" not in pdf_text_store:
        return jsonify({"error": "Please upload a PDF first."}), 400

    context = "\n\n---\n\n".join(find_relevant_chunks(pdf_text_store["current"], question))
    prompt = f"""Answer the question based ONLY on the PDF content below.
If the answer is not found, say "I couldn't find that in the PDF."

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
    app.run(debug=True)
