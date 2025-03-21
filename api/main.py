# api/main.py

import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from triage_logic import triage_questions, generate_triage_recommendation
from nlp_inference import nlp_predict
from vision_inference import vision_predict

app = FastAPI(
    title="Hatme Medical Diagnosis AI",
    description="Prototype system that demonstrates text-based triage and image diagnosis.",
    version="1.0.0",
)

@app.get("/")
def root():
    return {"message": "Welcome to Hatme API"}

@app.post("/triage")
def triage(
    age: int = Form(...),
    appearance: str = Form(...),
    symptoms: str = Form(...),
    # ... additional fields from SIT DOWN SIR, WWHAM, etc.
):
    """
    Example triage endpoint that uses structured info from user input forms
    to perform question-based analysis. In a real scenario, you'd gather more
    data over multiple requests or a conversation flow.
    """
    # Simple logic: pass user inputs to triage system
    triage_result = generate_triage_recommendation(age, appearance, symptoms)
    # Potentially run it through NLP to refine suggestions
    nlp_diagnosis = nlp_predict(symptoms)
    return {
        "triage_result": triage_result,
        "nlp_diagnosis": nlp_diagnosis
    }

@app.post("/diagnose-image")
async def diagnose_image(file: UploadFile = File(...)):
    """
    Endpoint for uploading an image (X-ray, CT, MRI) for classification.
    """
    contents = await file.read()
    diagnosis = vision_predict(contents)
    return {"diagnosis": diagnosis}

if __name__ == "__main__":
    # If you want to run locally
    uvicorn.run(app, host="0.0.0.0", port=8000)
