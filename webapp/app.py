# webapp/app.py
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests

app = FastAPI(title="Hatme Web App")

# Mount static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Replace with your actual API base URL once deployed.
API_BASE_URL = "https://your-cloud-run-url"  

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/triage", response_class=HTMLResponse)
def triage_form(request: Request):
    return templates.TemplateResponse("triage.html", {"request": request})

@app.post("/triage", response_class=HTMLResponse)
async def triage_submit(request: Request, age: int = Form(...), appearance: str = Form(...), symptoms: str = Form(...)):
    # Send data to the API triage endpoint
    response = requests.post(
        f"{API_BASE_URL}/triage",
        data={"age": age, "appearance": appearance, "symptoms": symptoms},
    )
    if response.status_code == 200:
        result = response.json()
    else:
        result = {"triage_result": "Error processing your input", "nlp_diagnosis": ""}
    return templates.TemplateResponse("triage_result.html", {"request": request, "result": result})

@app.get("/diagnose", response_class=HTMLResponse)
def diagnose_form(request: Request):
    return templates.TemplateResponse("diagnose.html", {"request": request})

@app.post("/diagnose", response_class=HTMLResponse)
async def diagnose_submit(request: Request, file: UploadFile = File(...)):
    # Send uploaded file to the API diagnose-image endpoint
    contents = await file.read()
    files = {"file": (file.filename, contents, file.content_type)}
    response = requests.post(f"{API_BASE_URL}/diagnose-image", files=files)
    if response.status_code == 200:
        result = response.json()
    else:
        result = {"diagnosis": "Error processing the image."}
    return templates.TemplateResponse("diagnose_result.html", {"request": request, "result": result})
