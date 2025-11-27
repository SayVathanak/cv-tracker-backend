import os  
from dotenv import load_dotenv 
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from typing import List
from pydantic import BaseModel
import cloudinary
import cloudinary.uploader
import pdfplumber
import docx
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import io
import re
import platform
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from datetime import datetime
import httpx
from fastapi.responses import Response
import json
import google.generativeai as genai

load_dotenv()

# --- 1. SETUP API KEYS ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("No GEMINI_API_KEY found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})

# --- 2. CONFIGURATION ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    print("Running on Linux/Cloud - using default Tesseract path")

cloudinary.config( 
  cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"), 
  api_key = os.getenv("CLOUDINARY_API_KEY"), 
  api_secret = os.getenv("CLOUDINARY_SECRET"),
  secure = True
)

MONGO_URL = os.getenv("MONGO_URL")
client = AsyncIOMotorClient(MONGO_URL)
db = client.cv_tracking_db
collection = db.candidates

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db_client():
    await collection.create_index([("Name", "text"), ("Tel", "text"), ("School", "text"), ("Location", "text")])

# --- 3. TEXT EXTRACTION (Helper) ---
def _extract_text_sync(file_bytes: bytes, filename: str) -> str:
    filename = filename.lower()
    text = ""
    try:
        if filename.endswith('.pdf'):
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    extract = page.extract_text()
                    if extract: text += extract + "\n"
            if len(text.strip()) < 50: # Scanned PDF fallback
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
        print(f"Error reading file {filename}: {e}")
    return text

async def extract_text(file_bytes: bytes, filename: str) -> str:
    return await run_in_threadpool(_extract_text_sync, file_bytes, filename)

    # --- 4. AI PARSING LOGIC (The Brain) ---
def parse_cv_with_ai(text: str) -> dict:
    """
    Uses Gemini AI to extract structured data from CV text.
    """
    # This prompt contains ALL your logic (Sangkats, Cities, Experience, etc.)
    prompt = f"""
    You are an expert HR Data Extractor for Cambodian Candidates.
    Analyze the following CV text and extract the details into a JSON object.

    ### RULES FOR LOCATION:
    1. **Phnom Penh:**
       - Try to find the specific "Sangkat" and "Khan".
       - Format: "Sangkat [Name], Khan [Name]" (e.g., "Sangkat Teuk Thla, Khan Sen Sok").
       - If only Khan is found: "Khan [Name]".
       - If nothing specific found but mentions Phnom Penh: "Phnom Penh".
    2. **Kandal & Takeo (Special Cities):**
       - Look for cities like "Ta Khmau", "Kien Svay", "Doun Kaev", "Tram Kak".
       - Format: "City, Province" (e.g., "Ta Khmau, Kandal").
    3. **Other Provinces:**
       - Return ONLY the Province Name (e.g., "Siem Reap", "Battambang").

    ### RULES FOR OTHER FIELDS:
    - **Name:** Full name (Capitalize properly). Remove titles like Mr/Ms.
    - **Tel:** Format as '0xx xxx xxx' (e.g., 012 345 678).
    - **School:** Extract the most recent University (Use standard abbreviations: RUPP, ITC, NUM, PUC, AUPP, CamTech, etc.).
    - **Experience:** Summarize the last job title and company in < 15 words.
    - **Gender:** Detect Male/Female.
    - **BirthDate:** Extract Date of Birth. Convert to format 'DD-MM-YYYY' (e.g., 25-12-1999). If not found, return "N/A".

    ### CV TEXT TO ANALYZE:
    {text[:15000]} 
    """

    try:
        # 1. Ask Gemini
        response = model.generate_content(prompt)
        
        # 2. Clean JSON
        json_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(json_text)
        
        # 3. Safety Fallback (Ensure keys exist)
        # ADDED "BirthDate" HERE
        defaults = {
            "Name": "N/A", 
            "Tel": "N/A", 
            "Location": "N/A", 
            "School": "N/A", 
            "Experience": "N/A", 
            "Gender": "N/A",
            "BirthDate": "N/A" 
        }
        
        for k, v in defaults.items():
            if k not in data or not data[k]:
                data[k] = v
        return data

    except Exception as e:
        print(f"AI Parsing Error: {e}")
        # ADDED "BirthDate" HERE AS WELL
        return {
            "Name": "Manual Review Needed", 
            "Tel": "N/A", 
            "Location": "Error Parsing", 
            "School": "N/A", 
            "Experience": "N/A", 
            "Gender": "N/A",
            "BirthDate": "N/A"
        }

# --- 5. API ENDPOINTS ---

@app.post("/upload-cv")
async def upload_cv(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            content = await file.read()
            
            # 1. Cloudinary
            upload_result = cloudinary.uploader.upload(content, resource_type="auto", public_id=file.filename.split('.')[0])
            cv_url = upload_result.get("secure_url")
            
            # 2. Extract Text
            raw_text = await extract_text(content, file.filename)
            
            # 3. Parse with AI
            structured_data = parse_cv_with_ai(raw_text)
            
            # 4. Add Metadata
            structured_data.update({
                "file_name": file.filename, 
                "cv_url": cv_url, 
                "upload_date": datetime.now().isoformat(), 
                "locked": False
            })
            
            # 5. Save to DB
            query = {"Name": structured_data["Name"], "Tel": structured_data["Tel"]}
            
            # Logic: If AI fails (returns N/A for both), just insert. If it finds real data, check for duplicates.
            if structured_data["Tel"] == "N/A" and structured_data["Name"] == "N/A":
                 result = await collection.insert_one(structured_data)
                 structured_data["_id"] = str(result.inserted_id)
            else:
                existing = await collection.find_one(query)
                if existing:
                    await collection.update_one({"_id": existing["_id"]}, {"$set": structured_data})
                    structured_data["_id"] = str(existing["_id"])
                else:
                    result = await collection.insert_one(structured_data)
                    structured_data["_id"] = str(result.inserted_id)
            
            results.append(structured_data)
            
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results.append({"filename": file.filename, "status": f"Error: {str(e)}"})
            
    return {"status": f"Processed {len(results)} files", "details": results}

@app.get("/candidates")
async def get_candidates(page: int = Query(1, ge=1), limit: int = Query(20, le=100), search: str = Query(None)):
    query_filter = {}
    if search:
        search_regex = {"$regex": search, "$options": "i"}
        query_filter = {
            "$or": [
                {"Name": search_regex}, {"Tel": search_regex},
                {"School": search_regex}, {"Location": search_regex},
                {"Experience": search_regex}
            ]
        }

    total_count = await collection.count_documents(query_filter)
    skip = (page - 1) * limit
    cursor = collection.find(query_filter).sort("upload_date", -1).skip(skip).limit(limit)
    
    candidates = []
    async for candidate in cursor:
        candidate["_id"] = str(candidate["_id"])
        candidates.append(candidate)
        
    return {"data": candidates, "page": page, "limit": limit, "total": total_count}

@app.get("/cv/{candidate_id}")
async def get_candidate_cv(candidate_id: str):
    try:
        obj_id = ObjectId(candidate_id)
        candidate = await collection.find_one({"_id": obj_id})
        if not candidate or "cv_url" not in candidate: return Response(content="CV not found", status_code=404)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(candidate["cv_url"])
            response.raise_for_status()
        return Response(content=response.content, media_type=response.headers.get("Content-Type", "application/pdf"))
    except Exception as e:
        return Response(content=f"Error: {e}", status_code=500)

@app.delete("/candidates/{candidate_id}")
async def delete_candidate(candidate_id: str):
    try:
        obj_id = ObjectId(candidate_id)
        candidate = await collection.find_one({"_id": obj_id})
        if candidate and candidate.get("locked", False): return {"status": "Error: Candidate is locked"}
        await collection.delete_one({"_id": obj_id})
        return {"status": "Deleted successfully"}
    except Exception as e:
        return {"status": f"Error: {e}"}

class BulkDeleteRequest(BaseModel):
    candidate_ids: List[str]
    passcode: str = None

@app.post("/candidates/bulk-delete")
async def delete_bulk_candidates(request: BulkDeleteRequest):
    try:
        if len(request.candidate_ids) == 0:
            if request.passcode != "9994": return {"status": "error", "message": "Invalid passcode"}
            await collection.delete_many({"locked": {"$ne": True}})
            return {"status": "success", "deleted": "All unlocked"}
        else:
            object_ids = [ObjectId(cid) for cid in request.candidate_ids]
            result = await collection.delete_many({"_id": {"$in": object_ids}, "locked": {"$ne": True}})
            return {"status": "success", "deleted": result.deleted_count}
    except Exception as e:
        return {"status": f"Error: {e}"}

@app.put("/candidates/{candidate_id}")
async def update_candidate(candidate_id: str, updated_data: dict):
    try:
        if "_id" in updated_data: del updated_data["_id"]
        updated_data["last_modified"] = datetime.now().isoformat()
        await collection.update_one({"_id": ObjectId(candidate_id)}, {"$set": updated_data})
        return {"status": "Updated successfully"}
    except Exception as e:
        return {"status": f"Error: {e}"}

@app.put("/candidates/{candidate_id}/lock")
async def toggle_lock(candidate_id: str, request: dict):
    try:
        await collection.update_one({"_id": ObjectId(candidate_id)}, {"$set": {"locked": request.get("locked", False)}})
        return {"status": "success"}
    except Exception as e:
        return {"status": f"Error: {e}"}