from fastapi import FastAPI
from app.extractor import extract_resume_data

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Resume Extractor API Running"}