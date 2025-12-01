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

def parse_cv_with_ai(text: str) -> dict:
    """
    Uses Gemini AI to extract structured data from CV text.
    """
    prompt = f"""
    You are an expert HR Data Extractor for Cambodian Candidates.
    Analyze the following CV text and extract the details into a JSON object.

    ### RULES FOR LOCATION:
    1. **Phnom Penh:** Format as "Sangkat [Name], Khan [Name]" or "Khan [Name]".
    2. **Provinces:** Return "City, Province" or just "Province Name".

    ### RULES FOR NEW FIELDS:
    - **Position:** Extract the Job Title the candidate is applying for. Look for "Applying for...", "Subject: Application for...", "Objective", or a professional title under their name. If not mentioned, return "N/A".
    - **School:** Extract the HIGHEST education level. 
       - Priority 1: University/Institute name (e.g., RUPP, SETEC, NUM).
       - Priority 2: If no university found, extract High School name.

    ### RULES FOR OTHER FIELDS:
    - **Name:** Full name (Capitalize properly). Remove titles.
    - **Tel:** Extract phone number (0xx ...).
    - **Experience:** Summarize last job title and company (< 15 words).
    - **Gender:** Detect Male/Female.
    - **BirthDate:** Format 'DD-MM-YYYY'.

    ### CV TEXT TO ANALYZE:
    {text[:15000]} 
    """

    try:
        response = model.generate_content(prompt)
        json_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(json_text)
        
        # Updated Defaults to include "Position"
        defaults = {
            "Name": "N/A", 
            "Tel": "N/A", 
            "Location": "N/A", 
            "School": "N/A", 
            "Experience": "N/A", 
            "Gender": "N/A",
            "BirthDate": "N/A",
            "Position": "N/A" # <--- NEW FIELD
        }
        
        for k, v in defaults.items():
            if k not in data or not data[k]:
                data[k] = v
        return data

    except Exception as e:
        print(f"AI Parsing Error: {e}")
        return {
            "Name": "Manual Review Needed", 
            "Tel": "N/A", 
            "Location": "Error Parsing", 
            "School": "N/A", 
            "Experience": "N/A", 
            "Gender": "N/A",
            "BirthDate": "N/A",
            "Position": "N/A"
        }

# --- 5. API ENDPOINTS ---

@app.post("/upload-cv")
async def upload_cv(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            content = await file.read()
            
            # 1. Cloudinary
            # Clean the filename: remove anything that isn't a letter or number
            clean_name = re.sub(r'[^a-zA-Z0-9]', '_', file.filename.split('.')[0])
            upload_result = cloudinary.uploader.upload(content, resource_type="auto", public_id=clean_name)
            
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
    
@app.post("/candidates/{candidate_id}/retry")
async def retry_parsing(candidate_id: str):
    try:
        # 1. Get Candidate
        obj_id = ObjectId(candidate_id)
        candidate = await collection.find_one({"_id": obj_id})
        
        if not candidate or "cv_url" not in candidate:
            return {"status": "error", "message": "Candidate or CV URL not found"}

        # 2. Download File from Cloudinary
        async with httpx.AsyncClient() as client:
            response = await client.get(candidate["cv_url"])
            if response.status_code != 200:
                return {"status": "error", "message": "Failed to download CV file"}
            file_bytes = response.content

        # 3. Extract Text (Reuse existing logic)
        # We assume file extension from the stored file_name or url
        file_name = candidate.get("file_name", "unknown.pdf")
        raw_text = await extract_text(file_bytes, file_name)

        # 4. Parse with AI (Reuse existing logic)
        structured_data = parse_cv_with_ai(raw_text)

        # 5. Merge & Update (Keep existing ID and Metadata, overwrite fields)
        update_payload = {
            "Name": structured_data["Name"],
            "Tel": structured_data["Tel"],
            "Location": structured_data["Location"],
            "School": structured_data["School"],
            "Experience": structured_data["Experience"],
            "Gender": structured_data["Gender"],
            "BirthDate": structured_data["BirthDate"],
            "Position": structured_data.get("Position", "N/A"), 
            "last_modified": datetime.now().isoformat()
        }

        await collection.update_one({"_id": obj_id}, {"$set": update_payload})

        return {"status": "success", "data": update_payload}

    except Exception as e:
        print(f"Retry Error: {e}")
        return {"status": "error", "message": str(e)}