from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import firebase_admin
from firebase_admin import credentials, firestore, storage
from openai import AzureOpenAI
import chromadb
from PyPDF2 import PdfReader
from docx import Document
from pptx import Presentation
from dotenv import load_dotenv
import tempfile

# Load .env
load_dotenv()

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Firebase configuration from environment variables
firebase_config = {
    "type": "service_account",
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
    "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL"),
    "universe_domain": "googleapis.com"
}

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred, {
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET")
    })

db = firestore.client()
bucket = storage.bucket()

# Initialize Azure OpenAI
azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Initialize Chroma Cloud
chroma_client = chromadb.CloudClient(
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE"),
    api_key=os.getenv("CHROMA_API_KEY")
)
collection = chroma_client.get_or_create_collection("documents")

# --- Text Extraction ---
def extract_text(file_path, ext):
    if ext == "pdf":
        reader = PdfReader(file_path)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    elif ext == "docx":
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == "pptx":
        prs = Presentation(file_path)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text
    return ""

# --- Chunking ---
def chunk_text(text, size=1000, overlap=200):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size - overlap)]

# --- Summarize ---
def summarize(text):
    chunks = chunk_text(text, size=12000, overlap=0)
    partial_summaries = []

    # Step 1: Summarize each chunk separately
    for i, chunk in enumerate(chunks):
        prompt = f"Summarize the following portion of a document in detail (part {i+1}/{len(chunks)}):\n\n{chunk}"
        resp = azure_client.chat.completions.create(
            model=os.getenv("AZURE_DEPLOYMENT_NAME"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000
        )
        part_summary = resp.choices[0].message.content.strip()
        partial_summaries.append(part_summary)

    # Step 2: Merge partial summaries into a cohesive summary
    combined_summary_text = "\n\n".join(partial_summaries)
    final_prompt = f"Combine the following section summaries into a cohesive, detailed overall summary of the entire document:\n\n{combined_summary_text}"

    final_resp = azure_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0,
        max_tokens=1200
    )

    return final_resp.choices[0].message.content.strip()

# --- Embed ---
def embed(text):
    resp = azure_client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return resp.data[0].embedding

# --- Routes ---
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""

    if not ext:
        return jsonify({"error": "File must have an extension"}), 400

    try:
        # --- Save to temp ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            file.save(tmp.name)
            text = extract_text(tmp.name, ext)

        if not text.strip():
            return jsonify({"error": "Could not extract text from file"}), 400

        # --- Upload to Firebase Storage ---
        blob = bucket.blob(f"docs/{file.filename}")
        blob.upload_from_filename(tmp.name)
        blob.make_public()
        url = blob.public_url

        # --- Summarize (now handles multiple chunks) ---
        summary = summarize(text)

        # --- Create Firestore doc ---
        doc_ref = db.collection("documents").add({
            "name": file.filename,
            "url": url,
            "summary": summary,
            "indexed": False,
            "uploadedAt": firestore.SERVER_TIMESTAMP
        })[1]

        # --- Index in Chroma ---
        for i, chunk in enumerate(chunk_text(text)):
            emb = embed(chunk)
            collection.add(
                ids=[f"{doc_ref.id}_{i}"],
                documents=[chunk],
                metadatas=[{"doc_id": doc_ref.id, "chunk_id": i}],
                embeddings=[emb]
            )

        # --- Mark as indexed ---
        doc_ref.update({"indexed": True})

        return jsonify({
            "id": doc_ref.id,
            "name": file.filename,
            "url": url,
            "summary": summary,
            "indexed": True
        })

    except Exception as e:
        print("❌ Upload error:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temp file
        try:
            os.remove(tmp.name)
        except Exception:
            pass

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")
    doc_id = data.get("doc_id")
    summary = data.get("summary")

    if not all([question, doc_id, summary]):
        return jsonify({"error": "Missing data"}), 400

    # Get embedding
    q_emb = embed(question)

    # Query Chroma
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=4,
        where={"doc_id": doc_id},
        include=["documents"]
    )
    context = "\n\n".join(results["documents"][0])

    # RAG Prompt
    prompt = f"Context:\n{context}\n\nSummary:\n{summary}\n\nQuestion:\n{question}\n\nAnswer clearly and concisely."

    resp = azure_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )

    return jsonify({"answer": resp.choices[0].message.content})

@app.route("/docs", methods=["GET"])
def list_docs():
    docs = []
    for doc in db.collection("documents").stream():
        data = doc.to_dict()
        data["id"] = doc.id
        docs.append(data)
    return jsonify(docs)

@app.route("/api/analyze-sentiment", methods=["POST"])
def analyze_sentiment_batch():
    data = request.get_json()
    texts = data.get("texts", [])
    
    if not texts:
        return jsonify([]), 200

    results = []
    for text in texts:
        try:
            # Use JSON mode to force valid output
            prompt = f"""Analyze the sentiment of this message. Return ONLY valid JSON:
{{
  "score": <number from -1.0 to 1.0>,
  "label": "positive" | "neutral" | "negative"
}}

Message: "{text.strip()}"

Return only the JSON."""
            
            response = azure_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=60
            )
            
            content = response.choices[0].message.content.strip()
            # Clean code blocks
            json_str = content.replace("```json", "").replace("```", "").strip()
            parsed = eval(json_str)  # safe due to response_format
            
            score = max(-1, min(1, float(parsed.get("score", 0))))
            label = parsed.get("label", "neutral")
            if label not in ["positive", "neutral", "negative"]:
                label = "neutral"
                
            results.append({"score": score, "label": label})
        except Exception as e:
            print(f"Sentiment error: {e}")
            results.append({"score": 0, "label": "neutral"})
    
    return jsonify(results)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
