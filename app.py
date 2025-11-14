import json
import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
import requests

import firebase_admin
from firebase_admin import credentials, firestore
from openai import AzureOpenAI

# --- Load environment ---
load_dotenv()

app = Flask(__name__)
CORS(app, origins="*", supports_credentials=True)  # Allow all origins

# ---- Azure OpenAI Client (for everything) ---
azure_client = AzureOpenAI(
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT")
)

# ---- Firebase for match-skills (Optional) ----
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase-adminsdk.json")
    firebase_admin.initialize_app(cred, {
        "storageBucket": "syncverse-b0d18.firebasestorage.app"
    })
db = firestore.client()

# ================= FILE EXTRACTORS =================

def extract_text_from_pdf(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        text = ""
        with open(tmp_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        os.unlink(tmp_path)
        return text.strip()
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_docx(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        doc = docx.Document(tmp_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        os.unlink(tmp_path)
        return text.strip()
    except Exception as e:
        return f"Error extracting text from DOCX: {str(e)}"

def extract_text_from_txt(url):
    try:
        response = requests.get(url)
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

def get_azure_response(messages, max_tokens=256, temperature=0.7):
    try:
        completion = azure_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            response_format={"type": "text"}
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error calling Azure OpenAI API: {str(e)}"

# ==================== SUMMARY ENDPOINT ====================

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
        summary_prompt = f"""
Please provide a short, concise summary of the main points of this document in as few words as possible.

Document: {file_name}
Content:
{extracted_text}

Give only the essential information in 2-3 sentences.
"""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides ultra-concise summaries of documents."
            },
            {
                "role": "user",
                "content": summary_prompt
            }
        ]
        summary = get_azure_response(messages, max_tokens=160, temperature=0.1)
        return jsonify({
            'summary': summary,
            'file_name': file_name,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# ==================== CHAT ENDPOINT ====================

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
        chat_prompt = f"""
Based only on the content below, answer the user's question as briefly as possible (one or two sentences).
If the answer isn't in the document, reply briefly that it can't be found.

Document: {file_name}
Content:
{extracted_text}

User Question: {question}

Give a short, direct answer.
"""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based only on document content. Be as brief and precise as possible."
            },
            {
                "role": "user",
                "content": chat_prompt
            }
        ]
        response = get_azure_response(messages, max_tokens=120, temperature=0.4)
        return jsonify({
            'response': response,
            'file_name': file_name,
            'question': question,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# ==================== HEALTH ENDPOINT ====================
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Flask server is running'})

# ============= SENTIMENT/MATCH-SKILLS ENDPOINTS ================
@app.route("/api/analyze-sentiment", methods=["POST"])
def analyze_sentiment_batch():
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
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=60
            )
            content = response.choices[0].message.content.strip()
            json_str = content.replace("``````", "").strip()
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

@app.route('/api/match-skills', methods=['POST'])
def match_skills():
    payload = request.get_json(silent=True) or {}
    project_desc = payload.get('projectDescription', '').strip()
    if not project_desc:
        return jsonify({"error": "projectDescription required"}), 400
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

    candidate_lines = [
        f"- {d['name']} ({d['email'].split('@')[0]})\n"
        f"  Skills: {', '.join([s.title() for s in d['skills']]) or 'None'}\n"
        f"  Experience: {d['experience']} years\n"
        f"  Bio: {d['bio'][:100]}{'...' if len(d['bio']) > 100 else ''}"
        for d in developers
    ]
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
        desc_lower = project_desc.lower()
        desc_words = set(desc_lower.split())
        fallback = []
        for d in developers:
            skill_hits = sum(1 for s in d['skills'] if s in desc_lower or s.split('.')[0] in desc_lower)
            exp_score = min(d['experience'], 10) * 0.06
            bio_hits = sum(1 for w in d['bio'].split() if w in desc_words) * 0.03
            score = min(0.98, (skill_hits * 0.25) + exp_score + bio_hits)
            if score >= 0.50:
                fallback.append({
                    "email": d['email'],
                    "score": round(score, 2),
                    "reason": f"{skill_hits} skill(s), {d['experience']}y exp"
                })
        top3 = sorted(fallback, key=lambda x: x["score"], reverse=True)[:3]
        return jsonify({"matches": top3})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
