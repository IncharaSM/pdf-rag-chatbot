import os
import json
import numpy as np
from numpy.linalg import norm
from dotenv import load_dotenv
load_dotenv()
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, render_template, send_file
from groq import Groq
#from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import io
import re

app = Flask(__name__)
client = Groq(api_key=os.environ.get("your_api_key"))

# Load embedding model once at startup
print("Loading embedding model...")
#embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model ready.")

pdf_text_store = {}       # { filename: { "chunks": [...], "embeddings": np.array } }
resume_store   = {}       # { "text": full resume text }
resume_chat_history = []  # list of { role, content }

# ── PDF helpers ────────────────────────────────────────────────

def extract_text_from_pdf(file):
    data = file.read()
    doc = fitz.open(stream=data, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_pdf_bytes(data):
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

'''def find_relevant_chunks_vector(question, top_n=4):
    query_vec = embed_model.encode([question], convert_to_numpy=True)[0]
    results = []
    for filename, data in pdf_text_store.items():
        for chunk, emb in zip(data["chunks"], data["embeddings"]):
            score = np.dot(query_vec, emb) / (norm(query_vec) * norm(emb) + 1e-10)
            results.append((score, filename, chunk))
    results.sort(reverse=True)
    return results[:top_n]'''

def find_relevant_chunks_vector(question, top_n=4):
    all_chunks, filenames = [], []
    for filename, data in pdf_text_store.items():
        for chunk in data["chunks"]:
            all_chunks.append(chunk)
            filenames.append(filename)

    if not all_chunks:
        return []

    vectorizer = TfidfVectorizer().fit(all_chunks + [question])
    chunk_vecs = vectorizer.transform(all_chunks)
    query_vec  = vectorizer.transform([question])
    scores     = cosine_similarity(query_vec, chunk_vecs)[0]

    top_indices = scores.argsort()[::-1][:top_n]
    return [(scores[i], filenames[i], all_chunks[i]) for i in top_indices]

# ── Existing PDF chatbot routes ────────────────────────────────

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
    print(f"Embedding {len(chunks)} chunks for '{file.filename}'...")
    #embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    print(f"Done embedding '{file.filename}'.")
    #pdf_text_store[file.filename] = {"chunks": chunks, "embeddings": embeddings}
    pdf_text_store[file.filename] = {"chunks": chunks}
    return jsonify({"message": f"'{file.filename}' loaded! {len(pdf_text_store)} PDF(s) total."})

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"pdf_count": len(pdf_text_store), "pdf_names": list(pdf_text_store.keys())})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided."}), 400
    if not pdf_text_store:
        return jsonify({"error": "Please upload a PDF first."}), 400
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

# ── Resume Analyzer routes ─────────────────────────────────────

@app.route("/analyze-resume", methods=["POST"])
def analyze_resume():
    global resume_store, resume_chat_history

    # Get resume text from uploaded PDF
    if "resume" not in request.files:
        return jsonify({"error": "No resume file received."}), 400

    resume_file = request.files["resume"]
    jd_text = request.form.get("jd", "").strip()

    try:
        resume_text = extract_text_from_pdf_bytes(resume_file.read())
    except Exception as e:
        return jsonify({"error": f"Failed to read resume PDF: {str(e)}"}), 500

    if not resume_text.strip():
        return jsonify({"error": "Resume PDF appears to be empty or scanned."}), 400

    # Store resume for chat session
    resume_store = {"text": resume_text}
    resume_chat_history = []

    # Build ATS analysis prompt
    if jd_text:
        prompt = f"""You are an expert ATS (Applicant Tracking System) analyzer.

Analyze this resume against the provided job description and return ONLY a valid JSON object with no extra text.

Resume:
{resume_text}

Job Description:
{jd_text}

Return this exact JSON structure:
{{
  "score": <integer 0-100>,
  "matched_keywords": [<list of keywords from JD found in resume>],
  "missing_keywords": [<list of important keywords from JD missing in resume>],
  "suggestions": [<list of 4-6 specific actionable suggestions to improve ATS score>],
  "summary": "<2 sentence overall assessment>"
}}"""
    else:
        prompt = f"""You are an expert ATS (Applicant Tracking System) analyzer.

Analyze this resume for general ATS compatibility and return ONLY a valid JSON object with no extra text.

Resume:
{resume_text}

Return this exact JSON structure:
{{
  "score": <integer 0-100 based on general ATS best practices>,
  "matched_keywords": [<list of strong keywords already present in the resume>],
  "missing_keywords": [<list of commonly expected keywords that are missing>],
  "suggestions": [<list of 4-6 specific actionable suggestions to improve ATS compatibility>],
  "summary": "<2 sentence overall assessment>"
}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.1
        )
        raw = response.choices[0].message.content.strip()

        # Clean up any markdown code fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse AI response. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": f"Groq API error: {str(e)}"}), 500


@app.route("/resume-chat", methods=["POST"])
def resume_chat():
    global resume_chat_history

    data = request.json
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "No message provided."}), 400
    if not resume_store:
        return jsonify({"error": "Please analyze a resume first."}), 400

    # Build system prompt with resume context
    system_prompt = f"""You are an expert resume writer and career coach helping to create an ATS-friendly resume.

The user's current resume text is:
---
{resume_store['text']}
---

Help the user improve their resume. When they ask you to rewrite or generate the final resume:
- Use clear section headers in ALL CAPS (e.g. SUMMARY, EXPERIENCE, EDUCATION, SKILLS)
- Use bullet points with dashes (-)
- Keep formatting simple and ATS-friendly (no tables, no columns)
- Be specific and use action verbs
- Include quantified achievements where possible"""

    # Add user message to history
    resume_chat_history.append({"role": "user", "content": message})

    # Build messages for API
    messages = [{"role": "system", "content": system_prompt}] + resume_chat_history

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=1024,
            temperature=0.3
        )
        reply = response.choices[0].message.content.strip()
        resume_chat_history.append({"role": "assistant", "content": reply})
        return jsonify({"answer": reply})
    except Exception as e:
        return jsonify({"error": f"Groq API error: {str(e)}"}), 500


@app.route("/generate-pdf", methods=["POST"])
def generate_pdf():
    data = request.json
    resume_text = data.get("text", "").strip()

    if not resume_text:
        return jsonify({"error": "No resume text provided."}), 400

    try:
        pdf = FPDF()
        pdf.set_margins(20, 20, 20)
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=20)

        for line in resume_text.split("\n"):
            line = line.rstrip()

            # Section headers (ALL CAPS lines)
            if line.isupper() and len(line.strip()) > 2:
                pdf.ln(3)
                pdf.set_font("Helvetica", style="B", size=12)
                pdf.set_text_color(30, 80, 160)
                pdf.cell(0, 8, line.strip(), new_x="LMARGIN", new_y="NEXT")
                pdf.set_draw_color(30, 80, 160)
                pdf.line(20, pdf.get_y(), 190, pdf.get_y())
                pdf.ln(2)
                pdf.set_text_color(0, 0, 0)

            # Bullet points
            elif line.strip().startswith("-"):
                pdf.set_font("Helvetica", size=10)
                pdf.set_text_color(0, 0, 0)
                bullet_text = "-  " + line.strip()[1:].strip()
                pdf.multi_cell(0, 6, bullet_text)

            # Empty lines
            elif line.strip() == "":
                pdf.ln(2)

            # Regular text
            else:
                pdf.set_font("Helvetica", size=10)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 6, line.strip())

        pdf_bytes = pdf.output()
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name="ATS_Resume.pdf"
        )

    except Exception as e:
        return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))