import os
import json
import re
import io
import numpy as np
from dotenv import load_dotenv
load_dotenv()
import fitz  # PyMuPDF
from flask import Flask, request, jsonify, render_template, send_file
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF

app = Flask(__name__)
client = Groq(api_key=os.environ.get("your_api_key"))

pdf_text_store      = {}   # { filename: { "chunks": [...] } }
resume_store        = {}   # { "text": full resume text }
resume_chat_history = []   # [ { role, content } ]

# ── PDF helpers ────────────────────────────────────────────────

def extract_text_from_pdf(file):
    data = file.read()
    doc  = fitz.open(stream=data, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_pdf_bytes(data):
    doc  = fitz.open(stream=data, filetype="pdf")
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

def find_relevant_chunks(question, top_n=4):
    """TF-IDF cosine similarity search across all uploaded PDFs."""
    all_chunks, filenames = [], []
    for filename, data in pdf_text_store.items():
        for chunk in data["chunks"]:
            all_chunks.append(chunk)
            filenames.append(filename)

    if not all_chunks:
        return []

    vectorizer  = TfidfVectorizer().fit(all_chunks + [question])
    chunk_vecs  = vectorizer.transform(all_chunks)
    query_vec   = vectorizer.transform([question])
    scores      = cosine_similarity(query_vec, chunk_vecs)[0]

    top_indices = scores.argsort()[::-1][:top_n]
    return [(scores[i], filenames[i], all_chunks[i]) for i in top_indices]

# ── PDF Chatbot routes ─────────────────────────────────────────

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
    pdf_text_store[file.filename] = {"chunks": chunks}

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
    data     = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "No question provided."}), 400
    if not pdf_text_store:
        return jsonify({"error": "Please upload a PDF first."}), 400

    top_chunks = find_relevant_chunks(question, top_n=4)
    all_chunks = [f"[From: {fname}]\n{chunk}" for _, fname, chunk in top_chunks]
    context    = "\n\n---\n\n".join(all_chunks)

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

    if "resume" not in request.files:
        return jsonify({"error": "No resume file received."}), 400

    resume_file = request.files["resume"]
    jd_text     = request.form.get("jd", "").strip()

    try:
        resume_text = extract_text_from_pdf_bytes(resume_file.read())
    except Exception as e:
        return jsonify({"error": f"Failed to read resume PDF: {str(e)}"}), 500

    if not resume_text.strip():
        return jsonify({"error": "Resume PDF appears to be empty or scanned."}), 400

    resume_store        = {"text": resume_text, "jd": jd_text}
    resume_chat_history = []

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
        raw    = response.choices[0].message.content.strip()
        raw    = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse AI response. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": f"Groq API error: {str(e)}"}), 500


@app.route("/resume-chat", methods=["POST"])
def resume_chat():
    global resume_chat_history

    data    = request.json
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "No message provided."}), 400
    if not resume_store:
        return jsonify({"error": "Please analyze a resume first."}), 400

    jd_section = f"""The target Job Description is:
---
{resume_store['jd']}
---
Tailor all suggestions and rewrites to match this job description.""" if resume_store.get('jd') else "No job description was provided — improve for general ATS compatibility."

    system_prompt = f"""You are an expert resume writer and career coach helping to create an ATS-friendly resume.

The user's current resume text is:
---
{resume_store['text']}
---

{jd_section}

When rewriting or improving the resume, follow these rules STRICTLY:
- Do NOT use any markdown formatting (no **, no *, no #, no backticks)
- Use plain text ONLY
- Section headers must be in ALL CAPS on their own line (e.g. SUMMARY, EXPERIENCE, EDUCATION, SKILLS)
- Use a dash (-) at the start of every bullet point line
- Separate sections with a blank line
- Keep formatting simple and ATS-friendly (no tables, no columns)
- Use strong action verbs and quantified achievements
- Do NOT truncate or cut off any content — write the complete resume"""

    resume_chat_history.append({"role": "user", "content": message})
    messages = [{"role": "system", "content": system_prompt}] + resume_chat_history

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            max_tokens=4096,
            temperature=0.3
        )
        reply = response.choices[0].message.content.strip()
        resume_chat_history.append({"role": "assistant", "content": reply})
        return jsonify({"answer": reply})
    except Exception as e:
        return jsonify({"error": f"Groq API error: {str(e)}"}), 500


@app.route("/generate-pdf", methods=["POST"])
def generate_pdf():
    data        = request.json
    resume_text = data.get("text", "").strip()

    if not resume_text:
        return jsonify({"error": "No resume text provided."}), 400

    try:
        # Strip markdown formatting
        resume_text = re.sub(r'\*\*(.*?)\*\*', r'\1', resume_text)  # **bold**
        resume_text = re.sub(r'\*(.*?)\*',     r'\1', resume_text)  # *italic*
        resume_text = re.sub(r'#{1,6}\s*',     '',    resume_text)  # # headers
        resume_text = re.sub(r'`{1,3}',        '',    resume_text)  # `code`
        resume_text = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', resume_text)  # _italic_
        # Step 1: aggressive unicode → ASCII sanitization
        replacements = {
            "\u2019": "'",  "\u2018": "'",  "\u2032": "'",
            "\u201c": '"',  "\u201d": '"',  "\u2033": '"',
            "\u2013": "-",  "\u2014": "-",  "\u2015": "-",
            "\u2022": "-",  "\u00b7": "-",  "\u25cf": "-",
            "\u2026": "...","\u00a0": " ",  "\u200b": "",
            "\u00e9": "e",  "\u00e8": "e",  "\u00ea": "e",
            "\u00e0": "a",  "\u00e2": "a",  "\u00f4": "o",
            "\u00fb": "u",  "\u00fc": "u",  "\u00e7": "c",
        }
        for k, v in replacements.items():
            resume_text = resume_text.replace(k, v)

        # Step 2: drop anything still outside latin-1 range
        resume_text = resume_text.encode("latin-1", errors="ignore").decode("latin-1")

        # Step 3: break any token longer than 50 chars (URLs, long strings)
        def safe_line(text):
            tokens = text.split(" ")
            out = []
            for tok in tokens:
                while len(tok) > 50:
                    out.append(tok[:50])
                    tok = tok[50:]
                out.append(tok)
            return " ".join(out)

        pdf = FPDF()
        pdf.set_margins(20, 20, 20)
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.set_font("Helvetica", size=10)

        for raw_line in resume_text.split("\n"):
            line = safe_line(raw_line.rstrip())
            stripped = line.strip()

            if not stripped:
                pdf.ln(3)
                continue

            # Section header: ALL CAPS, short, no special chars
            if (stripped.isupper()
                    and 2 < len(stripped) < 50
                    and not stripped.startswith("-")):
                pdf.ln(4)
                pdf.set_font("Helvetica", style="B", size=11)
                pdf.set_text_color(30, 80, 160)
                pdf.multi_cell(170, 7, stripped)
                pdf.set_draw_color(30, 80, 160)
                pdf.line(20, pdf.get_y(), 190, pdf.get_y())
                pdf.ln(2)
                pdf.set_font("Helvetica", size=10)
                pdf.set_text_color(0, 0, 0)

            # Bullet point
            elif stripped.startswith("-"):
                pdf.set_font("Helvetica", size=10)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(170, 6, "-  " + stripped[1:].strip())

            # Normal line
            else:
                pdf.set_font("Helvetica", size=10)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(170, 6, stripped)

        pdf_bytes = pdf.output()
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype="application/pdf",
            as_attachment=True,
            download_name="ATS_Resume.pdf"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))