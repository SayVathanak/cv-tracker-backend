from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles # <--- NEW: Needed for Preview
import pdfplumber
import docx
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import re
import os       # <--- NEW: To manage folders
import shutil   # <--- NEW: To save files
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from typing import List

# --- CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- CREATE UPLOAD FOLDER ---
# This creates a folder named "static_uploads" to keep PDF/Images safe
UPLOAD_DIR = "static_uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# --- DATABASE SETUP ---
MONGO_URL = "mongodb+srv://saksovathanaksay_db_user:Vathanak99@cluster0.pt9gimf.mongodb.net/?appName=Cluster0"
client = AsyncIOMotorClient(MONGO_URL)
db = client.cv_tracking_db
collection = db.candidates

app = FastAPI()

# --- MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MOUNT STATIC FILES ---
# This allows the Frontend to access http://.../static/cv.pdf
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

# --- TEXT EXTRACTION ---
def extract_text(file_bytes: bytes, filename: str) -> str:
    filename = filename.lower()
    text = ""
    try:
        if filename.endswith('.pdf'):
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    extract = page.extract_text()
                    if extract: text += extract + "\n"
            if len(text) < 50:
                images = convert_from_bytes(file_bytes)
                for img in images:
                    text += pytesseract.image_to_string(img, lang='khm+eng')
        elif filename.endswith('.docx'):
            doc = docx.Document(io.BytesIO(file_bytes))
            text = "\n".join([para.text for para in doc.paragraphs])
        elif filename.endswith(('.jpg', '.jpeg', '.png')):
            image = Image.open(io.BytesIO(file_bytes))
            text = pytesseract.image_to_string(image, lang='khm+eng')
    except Exception as e:
        print(f"Error reading file: {e}")
    return text

# --- PARSING LOGIC ---
def parse_cv_text(text: str) -> dict:
    data = {"Name": "N/A", "Birth": "N/A", "Tel": "N/A", "Location": "N/A", "School": "N/A", "Experience": "N/A"}
    
    name_match = re.search(r"(?:Name|ឈ្មោះ)[\s.:]*([^\n]+)", text, re.IGNORECASE)
    if name_match:
        data["Name"] = name_match.group(1).replace("Address", "").strip().lstrip(". ")
    else:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            if "resume" not in lines[0].lower(): data["Name"] = lines[0]

    phone_match = re.search(r"\(?\+?\d{1,4}\)?[\s-]?\d{2,4}[\s-]?\d{2,4}[\s-]?\d{0,4}", text)
    if phone_match and len(phone_match.group(0)) > 8: data["Tel"] = phone_match.group(0).strip()

    dob_match = re.search(r"(?:Birth|DOB).*?(\d{1,2}[\s/-]+[A-Za-z0-9]+[\s/-]+\d{4})", text, re.IGNORECASE)
    if dob_match: data["Birth"] = dob_match.group(1).strip()

    loc_label = re.search(r"(?:Address|Location).*?([A-Za-z0-9\s,]+(?:Province|City|Street|Road|Village|Commune))", text, re.IGNORECASE | re.DOTALL)
    if loc_label:
        data["Location"] = loc_label.group(1).replace("\n", " ").strip()
    else:
        cambodia_geo = re.search(r"([A-Za-z0-9\s,\-]+(?:Commune|Sangkat|District|Khan|Phnom Penh|Province|Kandal|Siem Reap))", text, re.IGNORECASE)
        if cambodia_geo: data["Location"] = cambodia_geo.group(1).replace("\n", " ").strip()

    school_match = re.search(r"(?:at|from)?\s*([A-Za-z\s&]+(?:Institute|University|College|High School))", text, re.IGNORECASE)
    if school_match: data["School"] = re.sub(r'\d+', '', school_match.group(1).replace("\n", " ").strip()).strip()

    exp_match = re.search(r"(?:Work Experience|History|Employment|Experiences)(.*?)(?:Education|Skills|Language|Reference|Personal|Marital|Ma\s*rital|Sex|Nationality|Address|Religion|Health|Height|Hobbies|$)", text, re.IGNORECASE | re.DOTALL)
    if exp_match:
        raw_exp = exp_match.group(1).strip()
        if re.search(r"^[:\-\s]*(No|None|N/A)", raw_exp, re.IGNORECASE):
            data["Experience"] = "No Experience Listed"
        elif len(raw_exp) > 5:
            clean_exp = re.split(r"(?:Religion|Health|Height|Hobbies|Marital|Ma\s*rital)", raw_exp, flags=re.IGNORECASE)[0]
            data["Experience"] = clean_exp[:50].lstrip(":.- ") + "..."
    
    return data

# --- API ENDPOINTS ---

@app.post("/upload-cv")
async def upload_cv(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            # 1. SAVE FILE TO DISK (Crucial for Preview)
            file_location = f"{UPLOAD_DIR}/{file.filename}"
            
            # Read content once
            content = await file.read()
            
            # Write content to disk
            with open(file_location, "wb+") as file_object:
                file_object.write(content)

            # 2. Extract & Parse
            raw_text = extract_text(content, file.filename)
            structured_data = parse_cv_text(raw_text)
            structured_data["file_name"] = file.filename
            
            # 3. Database Upsert
            existing = await collection.find_one({"Name": structured_data["Name"], "Tel": structured_data["Tel"]})
            if existing:
                await collection.update_one({"_id": existing["_id"]}, {"$set": structured_data})
                status = "Updated"
            else:
                await collection.insert_one(structured_data)
                status = "Saved"
            
            if "_id" in structured_data: structured_data["_id"] = str(structured_data["_id"])
            results.append(structured_data)
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({"filename": file.filename, "status": f"Error: {str(e)}"})

    return {"status": f"Batch Processed {len(results)} files", "details": results}

@app.get("/candidates")
async def get_candidates():
    candidates = []
    async for candidate in collection.find():
        candidate["_id"] = str(candidate["_id"])
        candidates.append(candidate)
    return candidates

@app.delete("/candidates/{candidate_id}")
async def delete_candidate(candidate_id: str):
    try:
        obj_id = ObjectId(candidate_id)
        await collection.delete_one({"_id": obj_id})
        return {"status": "Deleted successfully"}
    except Exception as e:
        return {"status": f"Error: {e}"}

@app.put("/candidates/{candidate_id}")
async def update_candidate(candidate_id: str, updated_data: dict):
    try:
        obj_id = ObjectId(candidate_id)
        if "_id" in updated_data: del updated_data["_id"]
        await collection.update_one({"_id": obj_id}, {"$set": updated_data})
        return {"status": "Updated successfully"}
    except Exception as e:
        return {"status": f"Error: {e}"}