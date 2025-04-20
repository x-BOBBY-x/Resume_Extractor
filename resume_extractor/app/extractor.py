import re
import pdfplumber
from typing import Dict, List
import spacy
from skill_list import skills as SKILL_LIST
import requests
import os

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "host.docker.internal")
OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11434")
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_resume_data(*args, **kwargs):
    raise NotImplementedError("Rule-based extraction has been removed. Use LLM extraction only.")

def extract_name(*args, **kwargs):
    raise NotImplementedError("Rule-based extraction has been removed.")

def extract_email(*args, **kwargs):
    raise NotImplementedError("Rule-based extraction has been removed.")

def extract_phone(*args, **kwargs):
    raise NotImplementedError("Rule-based extraction has been removed.")

def extract_skills(*args, **kwargs):
    raise NotImplementedError("Rule-based extraction has been removed.")

def extract_experience(*args, **kwargs):
    raise NotImplementedError("Rule-based extraction has been removed.")

def extract_resume_data_llm(text: str, api_key: str) -> dict:
    endpoint = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = f"""Extract the following fields from this resume: Name, Email, Phone, Skills (as a list), Experience (as a paragraph). Return as JSON.\nResume:\n{text}\n"""
    response = requests.post(endpoint, headers=headers, json={"inputs": prompt})
    result = response.json()
    import json
    try:
        return json.loads(result[0]['generated_text'])
    except Exception:
        return {"error": "LLM extraction failed", "raw": result}

def extract_resume_data_ollama(text_or_pdf: str, model: str = "mistral") -> dict:

    import os
    import re
    if os.path.isfile(text_or_pdf):
        import pdfplumber
        with pdfplumber.open(text_or_pdf) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    else:
        text = text_or_pdf
    prompt = (
        "You are an expert resume parser. Carefully extract the following fields ONLY from the provided resume text and return a valid JSON object with these exact keys: "
        "name, email, phone, skills (as a list of strings), experience (as a string). "
        "Always extract the most accurate information present. If a field is missing, use an empty string or empty list. "
        "Do NOT make up information, do NOT add explanations, do NOT return code. Output ONLY the JSON. "
        "Skills must be a list of actual skills mentioned in the resume. Experience should be a summary of work experience, not education.\n"
        "The email must be a valid email address (e.g., user@example.com). The phone must be a valid phone number (e.g., +1-123-456-7890 or 9876543210).\n\n"
        f"Resume:\n{text}\n"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=90)
        result = response.json()
        import json as pyjson
        match = re.search(r"\{.*\}", result['response'], re.DOTALL)
        if match:
            data = pyjson.loads(match.group(0))
            
            data['name'] = data.get('name', '').strip()

            email = data.get('email', '').strip()
            email_regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
            if not re.fullmatch(email_regex, email):
                data['email'] = ''
            else:
                data['email'] = email

            phone = data.get('phone', '').strip()
            phone_regex = r"(\+\d{1,3}[- ]?)?\d{10}|\(\d{3}\) ?\d{3}-\d{4}|\d{3}[- ]?\d{3}[- ]?\d{4}"
            if not re.fullmatch(phone_regex, phone):
                data['phone'] = ''
            else:
                data['phone'] = phone
            skills = data.get('skills', [])
            if isinstance(skills, str):
                import ast
                try:
                    skills = ast.literal_eval(skills)
                except Exception:
                    skills = [s.strip() for s in skills.split(",") if s.strip()]
            if not isinstance(skills, list):
                skills = [str(skills)]
            data['skills'] = [s.strip() for s in skills if s.strip()]
            data['experience'] = data.get('experience', '').strip()
            return data
        else:
            return {"error": "No JSON found in LLM response", "raw": result['response']}
    except Exception as e:
        return {"error": f"Ollama extraction failed: {e}", "raw": result if 'result' in locals() else None}

def extract_resume_data_hybrid(*args, **kwargs):
    raise NotImplementedError("Hybrid extraction has been removed. Use LLM extraction only.")
