import json
import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
import requests
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv

load_dotenv()

# --- Azure OpenAI config (REST) ---
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_DEPLOYMENT = os.environ.get("AZURE_DEPLOYMENT_NAME", "")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_VERSION", "2023-05-15")

if not (AZURE_ENDPOINT and AZURE_API_KEY and AZURE_DEPLOYMENT):
    raise RuntimeError("Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY and AZURE_DEPLOYMENT_NAME in your .env")

# --- Flask + CORS ---
app = Flask(__name__)
# allow all origins (change to specific origins in production)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Firebase Admin init (service account from env) ---
firebase_config = {
    "type": "service_account",
    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": (os.getenv("FIREBASE_PRIVATE_KEY") or "").replace('\\n', '\n'),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
    "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_CERT_URL"),
    "universe_domain": "googleapis.com"
}

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_config)
        initialize_app(cred)
    db = firestore.client()
except Exception as e:
    # If firebase fails to init, keep db = None and surface meaningful errors later
    print("Firebase init error:", e)
    db = None

# --- Text extraction functions (PDF/DOCX/TXT) ---
def extract_text_from_pdf(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        text = ""
        with open(temp_file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        os.unlink(temp_file_path)
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_docx(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        doc = docx.Document(temp_file_path)
        text = "\n".join([p.text for p in doc.paragraphs if p.text])
        os.unlink(temp_file_path)
        return text.strip()
    except Exception as e:
        return f"Error extracting text from DOCX: {str(e)}"

def extract_text_from_txt(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text.strip()
    except Exception as e:
        return f"Error extracting text from TXT: {str(e)}"

def extract_text_from_file(url, file_name):
    if file_name.lower().endswith('.pdf'):
        return extract_text_from_pdf(url)
    elif file_name.lower().endswith('.docx'):
        return extract_text_from_docx(url)
    elif file_name.lower().endswith('.txt'):
        return extract_text_from_txt(url)
    else:
        return f"Unsupported file type: {file_name}"

# --- Azure OpenAI REST helper ---
def get_azure_openai_response(messages, max_tokens=256, temperature=0.0):
    """
    Call Azure OpenAI Chat Completions (REST API).
    messages: list of {"role": "system|user|assistant", "content": "..."}
    """
    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    headers = {
        "api-key": AZURE_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Support typical response structure
        content = data["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        return f"Error calling Azure OpenAI: {str(e)}"

# --- Existing summary & chat endpoints (kept, using Azure) ---
@app.route('/summary', methods=['POST'])
def generate_summary():
    try:
        data = request.get_json()
        file_url = data.get('file_url')
        file_name = data.get('file_name')
        if not file_url or not file_name:
            return jsonify({'error': 'File URL and name are required'}), 400
        extracted_text = extract_text_from_file(file_url, file_name)
        if extracted_text.startswith("Error") or extracted_text.startswith("Unsupported"):
            return jsonify({'error': extracted_text}), 400
        if len(extracted_text) > 10000:
            extracted_text = extracted_text[:10000] + "... [text truncated]"

        system_message = (
            "You are an assistant that must answer ONLY from the provided document content. "
            "If the requested information is not present in the document, reply exactly: "
            "\"Cannot be found in the document.\" Do not ask follow-ups and do not provide "
            "any outside information."
        )
        user_prompt = (
            "Provide an ultra-concise summary (2-3 sentences max) of the main points below. "
            "Only use the document content and nothing else.\n\n"
            f"Document: {file_name}\n\nContent:\n{extracted_text}\n\n"
            "Return the summary as plain text."
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]

        summary = get_azure_openai_response(messages, max_tokens=200, temperature=0.2)
        if isinstance(summary, str) and "Cannot be found in the document." in summary:
            out_summary = "Cannot be found in the document."
        else:
            out_summary = summary.strip()

        return jsonify({'summary': out_summary, 'file_name': file_name, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat_with_document():
    try:
        data = request.get_json()
        file_url = data.get('file_url')
        file_name = data.get('file_name')
        question = data.get('question')
        if not file_url or not file_name or not question:
            return jsonify({'error': 'File URL, name, and question are required'}), 400
        extracted_text = extract_text_from_file(file_url, file_name)
        if extracted_text.startswith("Error") or extracted_text.startswith("Unsupported"):
            return jsonify({'error': extracted_text}), 400
        if len(extracted_text) > 8000:
            extracted_text = extracted_text[:8000] + "... [text truncated for processing]"

        system_message = (
            "You are an assistant that must answer ONLY from the provided document content. "
            "If the requested information is not present in the document, reply exactly: "
            "\"Cannot be found in the document.\" Answer in 1-2 sentences. Do not ask follow-ups or "
            "provide any outside information."
        )

        user_prompt = (
            "Based only on the content below, answer the user's question in one or two sentences. "
            "If the answer isn't in the document, reply exactly: \"Cannot be found in the document.\"\n\n"
            f"Document: {file_name}\n\nContent:\n{extracted_text}\n\n"
            f"User Question: {question}\n\nGive a short, direct answer."
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]

        response = get_azure_openai_response(messages, max_tokens=200, temperature=0.0)
        if isinstance(response, str) and "Cannot be found in the document." in response:
            out_response = "Cannot be found in the document."
        else:
            out_response = response.strip()

        return jsonify({'response': out_response, 'file_name': file_name, 'question': question, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# --- Added endpoint: analyze-sentiment (batch) ---
@app.route('/api/analyze-sentiment', methods=['POST'])
def analyze_sentiment_batch():
    data = request.get_json() or {}
    texts = data.get("texts", [])
    if not texts:
        return jsonify([]), 200

    results = []
    for text in texts:
        try:
            prompt = f"""Analyze the sentiment of this message. Return ONLY valid JSON like:
{{ "score": <number from -1.0 to 1.0>, "label": "positive" | "neutral" | "negative" }}
Message: {json.dumps(text)}
Return only the JSON object (no explanation)."""
            messages = [{"role": "user", "content": prompt}]
            content = get_azure_openai_response(messages, max_tokens=80, temperature=0.0)
            # strip possible code fences
            if isinstance(content, str):
                clean = content.replace("```json", "").replace("```", "").strip()
            else:
                clean = content
            parsed = json.loads(clean)
            score = float(parsed.get("score", 0))
            score = max(-1.0, min(1.0, score))
            label = parsed.get("label", "neutral")
            if label not in ["positive", "neutral", "negative"]:
                label = "neutral"
            results.append({"score": score, "label": label})
        except Exception as e:
            print("Sentiment parse error:", e)
            results.append({"score": 0.0, "label": "neutral"})
    return jsonify(results)

# --- Added endpoint: match-skills ---
@app.route('/api/match-skills', methods=['POST'])
def match_skills():
    payload = request.get_json(silent=True) or {}
    project_desc = payload.get('projectDescription', '').strip()
    if not project_desc:
        return jsonify({"error": "projectDescription required"}), 400

    if db is None:
        return jsonify({"error": "Firestore not initialized"}), 500

    # === 1. FETCH USERS FROM FIRESTORE ===
    try:
        users_snap = db.collection('users').stream()
        developers = []
        for doc in users_snap:
            data = doc.to_dict() or {}
            email = (data.get('email') or '').strip().lower()
            if not email:
                continue
            developers.append({
                "name": data.get('name') or email.split('@')[0],
                "email": email,
                "skills": [s.strip().lower() for s in (data.get('skills') or [])],
                "experience": int(data.get('experience') or 0),
                "bio": (data.get('bio') or '').lower()
            })
        if not developers:
            return jsonify({"matches": []}), 200
    except Exception as e:
        print("Firestore error:", e)
        return jsonify({"error": "Failed to load users"}), 500

    # === 2. BUILD CANDIDATE LINES ===
    candidate_lines = [
        f"- {d['name']} ({d['email'].split('@')[0]})\n"
        f"  Skills: {', '.join([s.title() for s in d['skills']]) or 'None'}\n"
        f"  Experience: {d['experience']} years\n"
        f"  Bio: {d['bio'][:100]}{'...' if len(d['bio']) > 100 else ''}"
        for d in developers
    ]

    # === 3. PROMPT & RULES ===
    prompt = f"""
Extract key technical skills and role level from this project:

\"{project_desc}\"

Then rank these candidates by fit.

CANDIDATES:
{chr(10).join(candidate_lines)}

RULES:
- Match exact and synonym skills (e.g., "React" = "React.js")
- Prefer higher experience for senior roles
- Use bio for context
- Score 0.00–1.00

RETURN ONLY JSON:
{{
  "matches": [
    {{ "email": "alice@ubti.com", "score": 0.94, "reason": "React + Firebase expert, 5y exp" }}
  ]
}}
Top 3 only. Score >= 0.50.
"""
    messages = [{"role": "user", "content": prompt}]
    try:
        content = get_azure_openai_response(messages, max_tokens=800, temperature=0.0)
        # remove fence markers if any
        if isinstance(content, str):
            clean = content.replace("```json", "").replace("```", "").strip()
        else:
            clean = content
        result = json.loads(clean)
        return jsonify(result)
    except Exception as e:
        print("LLM or parse error:", e)
        return jsonify({"error": "Failed to process skill matching"}), 500

# --- Health endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Flask server is running'})

if __name__ == '__main__':
    # In production, use a WSGI server (gunicorn/uvicorn) and set debug=False
    app.run(debug=True, host='0.0.0.0', port=5000)
