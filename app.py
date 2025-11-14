from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import requests
import os
from PyPDF2 import PdfReader
import docx
import tempfile
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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

def get_groq_response(messages, stream=False):
    """Get response from Groq API"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_completion_tokens=256,  # Short replies
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

@app.route('/summary', methods=['POST'])
def generate_summary():
    """Generate concise summary for a document"""
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
        summary = get_groq_response(messages)
        return jsonify({
            'summary': summary,
            'file_name': file_name,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat_with_document():
    """Chat with document content (brief answers)"""
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
        response = get_groq_response(messages)
        return jsonify({
            'response': response,
            'file_name': file_name,
            'question': question,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Flask server is running'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
