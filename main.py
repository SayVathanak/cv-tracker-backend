from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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

# --- CONFIGURATION ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    print("Running on Linux/Cloud - using default Tesseract path")

# --- CLOUDINARY SETUP ---
cloudinary.config( 
  cloud_name = "dsy9bfpre", 
  api_key = "973775943389468", 
  api_secret = "MW6-sD1o_2ck4-XTHIeoH8qEXO4",
  secure = True
)

# --- DATABASE SETUP ---
MONGO_URL = "mongodb+srv://saksovathanaksay_db_user:Vathanak99@cluster0.pt9gimf.mongodb.net/?appName=Cluster0"
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

# --- TEXT EXTRACTION ---
def extract_text(file_bytes: bytes, filename: str) -> str:
    filename = filename.lower()
    text = ""
    try:
        if filename.endswith('.pdf'):
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    extract = page.extract_text()
                    if extract: 
                        text += extract + "\n"
            if len(text.strip()) < 50:
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

# --- IMPROVED PARSING LOGIC ---
def parse_cv_text(text: str) -> dict:
    # 1. Initialization (Removed Email and Skills)
    data = {
        "Name": "N/A", "Birth": "N/A", "Tel": "N/A",
        "Location": "N/A", "School": "N/A", "Experience": "N/A",
        "Gender": "N/A"
    }
    
    text_normalized = re.sub(r'\s+', ' ', text)
    
    # 2. NAME EXTRACTION
    best_name = ""
    name_patterns = [
        r"(?:Name|Full\s*Name)[\s:.-]*([A-Z][a-zA-Z\s]{2,50}?)(?=\n|Address|Date|Tel|Contact|Mobile|$)",
        r"^([A-Z][a-zA-Z\s]{2,40})\n",
    ]
    for pattern in name_patterns:
        name_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if name_match:
            name = name_match.group(1).strip()
            exclude = ['resume', 'curriculum', 'contact', 'vitae', 'apply', 'summary', 'profile']
            if len(name) > 3 and not any(w in name.lower() for w in exclude) and not re.search(r'\d', name):
                best_name = name.split('\n')[0].strip()
                break
    if not best_name:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if lines and len(lines[0]) < 50 and "resume" not in lines[0].lower():
            best_name = lines[0]
    data["Name"] = best_name if best_name else "N/A"

    # 3. PHONE (Includes fix for spaced numbers like 096 51 35 387)
    phone_match = re.search(r'(?:\+855|0)(?:[\s-]*\d){8,11}', text)
    if phone_match:
        digits = re.sub(r'\D', '', phone_match.group(0))
        if digits.startswith('855'): digits = '0' + digits[3:]
        data["Tel"] = digits[:10]

    # 4. BIRTH DATE
    dob_match = re.search(r"(?:Birth|DOB).*?(\d{1,2}[-/\s\.]\d{1,2}[-/\s\.]\d{2,4})", text, re.IGNORECASE | re.DOTALL)
    if dob_match: data["Birth"] = dob_match.group(1).replace('\n', '').strip()

    # 5. LOCATION
    loc_match = re.search(r'(?:Address|Location)[\s:.-]*([^\n]{5,100})', text, re.IGNORECASE)
    if loc_match: data["Location"] = loc_match.group(1).strip()[:100]

    # 6. SCHOOL
    school_match = re.search(r'([A-Z][A-Za-z\s&-]+(?:University|Institute|High\s*School))', text)
    if school_match: data["School"] = school_match.group(1).strip()

    # 7. EXPERIENCE (Fixed to stop at Languages/Personalities)
    exp_match = re.search(
        r'(?:WORK\s+)?(?:EXPERIENCE|EMPLOYMENT|HISTORY)[\s:.-]*'  # Start looking here
        r'(.*?)'                                                   # Capture content
        r'(?=\n(?:EDUCATION|SKILLS|REFERENCE|LANGUAGES?|PERSONALITIES|INTEREST|DECLARATION|CERTIF|$))', # STOP here
        text, 
        re.IGNORECASE | re.DOTALL
    )
    
    if exp_match: 
        raw = re.sub(r'\s+', ' ', exp_match.group(1)).strip()
        # Clean up specific noise seen in your screenshot (like "S: . |")
        raw = re.sub(r'^[|:.\s]*', '', raw) 
        data["Experience"] = raw[:350] + ("..." if len(raw) > 350 else "")
        
    # 8. GENDER (Robust Fallback)
    found_gender = False
    explicit_match = re.search(r'\b(?:Gender|Sex)[\s:.-]*([A-Za-z]+)\b', text, re.IGNORECASE)
    if explicit_match:
        val = explicit_match.group(1).lower()
        if 'female' in val or 'f' == val: 
            data["Gender"] = "Female"
            found_gender = True
        elif 'male' in val or 'm' == val: 
            data["Gender"] = "Male"
            found_gender = True
            
    if not found_gender:
        if re.search(r'\bFemale\b', text, re.IGNORECASE):
            data["Gender"] = "Female"
        elif re.search(r'\bMale\b', text, re.IGNORECASE):
            data["Gender"] = "Male"

    return data

# --- API ENDPOINTS ---

@app.post("/upload-cv")
async def upload_cv(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            content = await file.read()
            
            upload_result = cloudinary.uploader.upload(
                content, 
                resource_type="auto", 
                public_id=file.filename.split('.')[0]
            )
            cv_url = upload_result.get("secure_url")

            raw_text = extract_text(content, file.filename)
            structured_data = parse_cv_text(raw_text)
            
            structured_data["file_name"] = file.filename
            structured_data["cv_url"] = cv_url
            structured_data["upload_date"] = datetime.now().isoformat()
            structured_data["locked"] = False
            
            query = {"Name": structured_data["Name"], "Tel": structured_data["Tel"]}
            
            existing = await collection.find_one(query)
            
            if existing:
                await collection.update_one(
                    {"_id": existing["_id"]}, 
                    {"$set": structured_data}
                )
                status = "Updated"
                structured_data["_id"] = str(existing["_id"])
            else:
                result = await collection.insert_one(structured_data)
                status = "Created"
                structured_data["_id"] = str(result.inserted_id)
            
            results.append(structured_data)
            
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results.append({"filename": file.filename, "status": f"Error: {str(e)}"})

    return {"status": f"Processed {len(results)} files", "details": results}

@app.get("/candidates")
async def get_candidates():
    candidates = []
    async for candidate in collection.find().sort("upload_date", -1):
        candidate["_id"] = str(candidate["_id"])
        candidates.append(candidate)
    return candidates

@app.get("/cv/{candidate_id}")
async def get_candidate_cv(candidate_id: str):
    try:
        obj_id = ObjectId(candidate_id)
        candidate = await collection.find_one({"_id": obj_id})

        if not candidate or "cv_url" not in candidate:
            return Response(content="CV not found", status_code=404)

        cv_url = candidate["cv_url"]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(cv_url)
            response.raise_for_status()

        content_type = response.headers.get("Content-Type", "application/pdf")
        return Response(content=response.content, media_type=content_type)
        
    except httpx.HTTPStatusError as e:
        print(f"Error fetching file from Cloudinary: {e}")
        return Response(content="Error fetching file from Cloudinary", status_code=500)
    except Exception as e:
        print(f"Internal error: {e}")
        return Response(content=f"Internal error: {e}", status_code=500)

@app.delete("/candidates/{candidate_id}")
async def delete_candidate(candidate_id: str):
    try:
        obj_id = ObjectId(candidate_id)
        candidate = await collection.find_one({"_id": obj_id})
        if candidate and candidate.get("locked", False):
            return {"status": "Error: Candidate is locked"}
        
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
        # Check if this is a "delete all" request (empty list means delete all)
        is_delete_all = len(request.candidate_ids) == 0
        
        # If deleting all, require passcode
        if is_delete_all:
            if request.passcode != "9994":
                return {
                    "status": "error",
                    "message": "Invalid passcode. Passcode required to delete all candidates."
                }
        
        deleted_count = 0
        skipped = []
        
        # Get candidates to delete
        if is_delete_all:
            # Delete all unlocked candidates
            candidates_cursor = collection.find({"locked": {"$ne": True}})
            async for candidate in candidates_cursor:
                await collection.delete_one({"_id": candidate["_id"]})
                deleted_count += 1
            
            # Count skipped (locked) candidates
            skipped_count = await collection.count_documents({"locked": True})
        else:
            # Delete specific candidates
            for cid in request.candidate_ids:
                obj_id = ObjectId(cid)
                candidate = await collection.find_one({"_id": obj_id})
                
                if candidate and candidate.get("locked", False):
                    skipped.append(cid)
                    continue
                
                await collection.delete_one({"_id": obj_id})
                deleted_count += 1
            
            skipped_count = len(skipped)
        
        return {
            "status": "success",
            "deleted": deleted_count,
            "skipped": skipped_count,
            "skipped_ids": skipped if not is_delete_all else []
        }
    except Exception as e:
        return {"status": f"Error: {e}"}

@app.put("/candidates/{candidate_id}")
async def update_candidate(candidate_id: str, updated_data: dict):
    try:
        obj_id = ObjectId(candidate_id)
        if "_id" in updated_data:
            del updated_data["_id"]
        updated_data["last_modified"] = datetime.now().isoformat()
        await collection.update_one({"_id": obj_id}, {"$set": updated_data})
        return {"status": "Updated successfully"}
    except Exception as e:
        return {"status": f"Error: {e}"}

@app.put("/candidates/{candidate_id}/lock")
async def toggle_lock(candidate_id: str, request: dict):
    try:
        obj_id = ObjectId(candidate_id)
        locked = request.get("locked", False)
        await collection.update_one(
            {"_id": obj_id}, 
            {"$set": {"locked": locked}}
        )
        return {"status": "success", "locked": locked}
    except Exception as e:
        return {"status": f"Error: {e}"}