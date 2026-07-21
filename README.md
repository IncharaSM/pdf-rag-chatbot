# 📄 PDF RAG Chatbot

A lightweight RAG (Retrieval-Augmented Generation) chatbot that lets you upload multiple PDFs and ask questions across all of them — powered by **Groq API**, **LLaMA 3.3** and **vector similarity search**.

![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Flask](https://img.shields.io/badge/Flask-2.x-green) ![Groq](https://img.shields.io/badge/Groq-LLaMA3.3-orange) ![Embeddings](https://img.shields.io/badge/Embeddings-sentence--transformers-purple)

## ✨ Features

- Upload multiple PDFs and extract text instantly
- Ask natural language questions across all uploaded documents
- Vector similarity search using `all-MiniLM-L6-v2` — understands meaning, not just keywords
- Per-PDF context retrieval with filename labels so the AI knows which document it's reading
- Fast responses via Groq's LLaMA 3.3-70b model
- Clean dark-mode UI

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/IncharaSM/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Groq API key
Get a free key at [console.groq.com](https://console.groq.com)

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

### 4. Run the app
```bash
python app.py
```

Open your browser at `http://localhost:5000`

> **Note:** On first run, the embedding model (`all-MiniLM-L6-v2`, ~90MB) will be downloaded automatically and cached locally.

## 🛠️ How It Works

1. **PDF Upload** — PyMuPDF extracts raw text from each uploaded PDF
2. **Chunking** — Text is split into ~3000 character chunks
3. **Embedding** — Every chunk is converted into a vector using `sentence-transformers` (`all-MiniLM-L6-v2`)
4. **Retrieval** — Your question is also embedded into a vector and matched against all chunks using **cosine similarity** — so semantically similar content is found even if the exact words differ
5. **Generation** — Top matching chunks (labeled by filename) + your question are sent to Groq's LLaMA 3.3 model
6. **Answer** — Response is displayed in the chat UI

## 📸 Screenshot

![UI Screenshot](image.png)

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| PDF Parsing | PyMuPDF (fitz) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| LLM | Groq API (LLaMA 3.3-70b-versatile) |
| Frontend | Vanilla HTML / CSS / JS |
| Deployment | Render |

## 📁 Project Structure

```
pdf-rag-chatbot/
├── app.py               # Flask backend — upload, embed, retrieve, answer
├── templates/
│   └── index.html       # Frontend UI
├── .env                 # API key (never commit this)
├── .gitignore
└── requirements.txt
```

## 🔮 Future Improvements

- [x] Support multiple PDFs
- [x] Swap keyword retrieval for vector similarity search
- [x] Deploy to Render
- [ ] Add conversation memory / chat history
- [ ] Persist PDFs across server restarts (database storage)
- [ ] Support scanned PDFs via OCR

## 👩‍💻 Author

**Inchara S M** — [LinkedIn](https://www.linkedin.com/in/inchara-s-m-459aa11a0) | [GitHub](https://github.com/IncharaSM)