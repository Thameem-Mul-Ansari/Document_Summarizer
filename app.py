import json
import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore, initialize_app
from openai import AzureOpenAI
from dotenv import load_dotenv
import requests
from PyPDF2 import PdfReader
import docx
from groq import Groq

# Load environment variables from .env
load_dotenv()

# Initialize Flask
app = Flask(__name__)

# Configure CORS to allow all origins and handle preflight requests
CORS(app, resources={
    r"/*": {
        "origins": "*",  # Allow all origins
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    }
})

# Add CORS headers for all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Handle OPTIONS method for preflight requests
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({"status": "preflight"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response, 200

# Initialize Firebase
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

if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_config)
    initialize_app(cred)

# Firestore and Firebase Storage clients
db = firestore.client()

# Initialize Groq Client for document processing
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize Azure OpenAI
azure_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# --- Text Extraction Functions ---
def extract_text_from_pdf(url):
    """Extract text from PDF file"""
    try:
        response = requests.get(url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        text = ""
        with open(temp_file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        os.unlink(temp_file_path)
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_docx(url):
    """Extract text from DOCX file"""
    try:
        response = requests.get(url)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        doc = docx.Document(temp_file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        os.unlink(temp_file_path)
        return text.strip()
    except Exception as e:
        return f"Error extracting text from DOCX: {str(e)}"

def extract_text_from_txt(url):
    """Extract text from TXT file"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text.strip()
    except Exception as e:
        return f"Error extracting text from TXT: {str(e)}"

def extract_text_from_file(url, file_name):
    """Extract text based on file type"""
    if file_name.lower().endswith('.pdf'):
        return extract_text_from_pdf(url)
    elif file_name.lower().endswith('.docx'):
        return extract_text_from_docx(url)
    elif file_name.lower().endswith('.txt'):
        return extract_text_from_txt(url)
    else:
        return f"Unsupported file type: {file_name}"

# --- Groq Response Handling ---
def get_groq_response(messages, stream=False):
    """Get response from Groq API"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_completion_tokens=256,
            top_p=1,
            stream=stream,
            stop=None
        )
        if stream:
            response_text = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
            return response_text
        else:
            return completion.choices[0].message.content
    except Exception as e:
        return f"Error calling Groq API: {str(e)}"

# --- Analyze Sentiment ---
@app.route('/api/analyze-sentiment', methods=['POST', 'OPTIONS'])
def analyze_sentiment_batch():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
        
    data = request.get_json()
    texts = data.get("texts", [])
    if not texts:
        return jsonify([]), 200

    results = []
    for text in texts:
        try:
            prompt = f"""Analyze the sentiment of this message. Return ONLY valid JSON:
{{
  "score": <number from -1.0 to 1.0>,
  "label": "positive" | "neutral" | "negative"
}}

Message: "{text.strip()}"

Return only the JSON."""
            response = azure_client.chat.completions.create(
                model="gpt-4o",  # cheaper & fast
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=60
            )
            content = response.choices[0].message.content.strip()
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

# --- Match Skills ---
@app.route('/api/match-skills', methods=['POST', 'OPTIONS'])
def match_skills():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
        
    payload = request.get_json(silent=True) or {}
    project_desc = payload.get('projectDescription', '').strip()

    if not project_desc:
        return jsonify({"error": "projectDescription required"}), 400

    # === 1. FETCH USERS FROM FIRESTORE ===
    try:
        users_snap = db.collection('users').stream()
        developers = []
        for doc in users_snap:
            data = doc.to_dict()
            email = data.get('email', '').strip().lower()
            if not email:
                continue
            developers.append({
                "name": data.get('name', email.split('@')[0]),
                "email": email,
                "skills": [s.strip().lower() for s in data.get('skills', [])] if data.get('skills') else [],
                "experience": int(data.get('experience', 0) or 0),
                "bio": (data.get('bio') or '').lower()
            })
        if not developers:
            return jsonify({"matches": []}), 200
    except Exception as e:
        print("Firestore error:", e)
        return jsonify({"error": "Failed to load users"}), 500

    # === 2. BUILD CANDIDATE LIST ===
    candidate_lines = [
        f"- {d['name']} ({d['email'].split('@')[0]})\n"
        f"  Skills: {', '.join([s.title() for s in d['skills']]) or 'None'}\n"
        f"  Experience: {d['experience']} years\n"
        f"  Bio: {d['bio'][:100]}{'...' if len(d['bio']) > 100 else ''}"
        for d in developers
    ]

    # === 3. CONCISE, SMART PROMPT ===
    prompt = f"""
Extract key technical skills and role level from this project:

"{project_desc}"

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

    # === 4. CALL AZURE OPENAI ===
    try:
        response = azure_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return jsonify(result)

    except Exception as e:
        print("LLM error:", e)
        # === 5. Fallback error handling ===
        return jsonify({"error": "Failed to process skill matching"}), 500

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET', 'OPTIONS'])
def health_check():
    """Health check endpoint"""
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    return jsonify({'status': 'healthy', 'message': 'Flask server is running'})

# Add a summary endpoint if needed
@app.route('/summary', methods=['POST', 'OPTIONS'])
def summary():
    if request.method == "OPTIONS":
        return jsonify({"status": "preflight"}), 200
    # Add your summary logic here
    return jsonify({"message": "Summary endpoint"})

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
