from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
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
    # 1. Initialization
    data = {
        "Name": "N/A", "Birth": "N/A", "Tel": "N/A",
        "Location": "N/A", "School": "N/A", "Experience": "N/A",
        "Gender": "N/A"
    }
    
    # Clean text
    text_normalized = re.sub(r'\s+', ' ', text).strip()
    text_lower = text_normalized.lower()
    
    # 1. NAME EXTRACTION (Improved)
    best_name = ""
    
    # Cut off text after "Reference" section to prevent grabbing reference names
    # This prevents "Voeun Vattanak" from being picked up
    text_for_name = text
    ref_match = re.search(r'\n(REFERENCE|REFERENCES)', text, re.IGNORECASE)
    if ref_match:
        text_for_name = text[:ref_match.start()]

    name_patterns = [
        # Pattern A: "Name: ..." (Classic)
        r"(?:Name|Full\s*Name)[\s:.-]*([A-Z][a-zA-Z\s.]{2,50}?)(?=\n|Address|Date|Tel|Contact|Mobile|$)",
        
        # Pattern B: Top of file (Now allows dots for Ms./Mr.)
        r"^([A-Z][a-zA-Z\s.]{2,40})(?:\n|$)",
    ]
    
    for pattern in name_patterns:
        # We only search in the SAFE text (before References)
        name_match = re.search(pattern, text_for_name, re.IGNORECASE | re.MULTILINE)
        if name_match:
            name = name_match.group(1).strip()
            # Bad words filter
            exclude = ['resume', 'curriculum', 'contact', 'vitae', 'apply', 'summary', 'profile', 'personal']
            if len(name) > 3 and not any(w in name.lower() for w in exclude) and not re.search(r'\d', name):
                # Clean up "Ms." or "Mr." prefix if you want just the name
                clean_name = re.sub(r'^(Ms\.|Mr\.|Mrs\.|Dr\.)\s*', '', name, flags=re.IGNORECASE)
                best_name = clean_name
                break
    
    # Fallback: Just grab the very first line if it looks like a name
    if not best_name:
        first_line = text.split('\n')[0].strip()
        if 3 < len(first_line) < 50 and "resume" not in first_line.lower():
            best_name = re.sub(r'^(Ms\.|Mr\.|Mrs\.)\s*', '', first_line, flags=re.IGNORECASE)

    data["Name"] = best_name if best_name else "N/A"

    # 3. PHONE
    phone_match = re.search(
        r'(?:(?:\+|00)?\s*\(?\s*(?:855|885)\s*\)?|0)\s*[-\s\.]?\s*[1-9][\d\s\-\.]{6,12}',
        text
    )

    if phone_match:
        digits = re.sub(r'\D', '', phone_match.group(0))
        if digits.startswith('885'): digits = '855' + digits[3:]
        if digits.startswith('855'): digits = '0' + digits[3:]
        elif digits.startswith('00855'): digits = '0' + digits[5:]
        if 9 <= len(digits) <= 10: data["Tel"] = digits

    # 4. BIRTH DATE
    dob_match = re.search(
        r"(?:Birth|DOB|Born)[\s:.-]*((?:\d{1,2}[-/\s\.]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/\s\.]+\d{4})|(?:\d{1,2}[-/\s\.]+\d{1,2}[-/\s\.]+\d{2,4}))", 
        text, re.IGNORECASE
    )
    if dob_match: 
        data["Birth"] = dob_match.group(1).strip()

    # 5. LOCATION (Khans & Provinces)
    found_location = False
    khan_map = {
        "Chamkar Mon": ["chamkar mon", "chamkarmon"], "Doun Penh": ["doun penh", "daun penh"],
        "7 Makara": ["7 makara", "prampir meakkakra"], "Toul Kork": ["tuol kouk", "toul kork"],
        "Dangkao": ["dangkao", "dangkor"], "Mean Chey": ["mean chey"], "Russey Keo": ["russey keo"],
        "Sen Sok": ["sen sok"], "Pou Senchey": ["pou senchey", "por sen chey"],
        "Chroy Changvar": ["chroy changvar"], "Prek Pnov": ["prek pnov"], "Chbar Ampov": ["chbar ampov"],
        "Boeung Keng Kang": ["boeng keng kang", "bkk"], "Kamboul": ["kamboul"]
    }
    for clean_name, variations in khan_map.items():
        pattern = r'\b(?:Khan|District)?[\s-]*(' + '|'.join(re.escape(v) for v in variations) + r')\b'
        if re.search(pattern, text, re.IGNORECASE):
            data["Location"] = clean_name; found_location = True; break

    if not found_location:
        province_map = {
            "Kampong Cham": ["kampong cham"], "Kampong Chhnang": ["kampong chhnang"],
            "Kampong Speu": ["kampong speu"], "Kampong Thom": ["kampong thom"], "Kandal": ["kandal"],
            "Takeo": ["takeo"], "Preah Vihear": ["preah vihear"], "Stung Treng": ["stung treng"],
            "Ratanakiri": ["ratanikiri"], "Siem Reap": ["siem reap"], "Oddar Meanchey": ["oddar meanchey"],
            "Banteay Meanchey": ["banteay meanchey"], "Battambang": ["battambang"], "Pursat": ["pursat"],
            "Pailin": ["pailin"], "Koh Kong": ["koh kong"], "Preah Sihanouk": ["preah sihanouk", "sihanoukville"],
            "Kampot": ["kampot"], "Kep": ["kep"], "Kratie": ["kratie"], "Mondulkiri": ["mondulkiri"],
            "Prey Veng": ["prey veng"], "Svay Rieng": ["svay rieng"], "Tbong Khmum": ["tbong khmum"],
            "Phnom Penh": ["phnom penh"]
        }
        for clean_name, variations in province_map.items():
            pattern = r'\b(?:Province|City|Krong|Khet)?[\s-]*(' + '|'.join(re.escape(v) for v in variations) + r')\b'
            if re.search(pattern, text, re.IGNORECASE):
                data["Location"] = clean_name; found_location = True; break

    if not found_location:
        loc_match = re.search(r'(?:Address|Location)[\s:.-]*([^\n]{5,100})', text, re.IGNORECASE)
        if loc_match: data["Location"] = loc_match.group(1).strip()[:100]

    # 6. SCHOOL
    uni_map = {
        "RUPP":   ["royal university of phnom penh", "rupp"],
        "ITC":    ["institute of technology of cambodia", "itc"],
        "RUA":    ["royal university of agriculture", "rua"],
        "RULE":   ["royal university of law and economics", "rule"],
        "NUM":    ["national university of management", "num"],
        "RUFA":   ["royal university of fine arts", "rufa"],
        "NPIC":   ["national polytechnic institute of cambodia", "npic"],
        "NPIA":   ["national polytechnic institute of angkor", "npia"],
        "NTTI": ["national technical training institute", "national training institute"],
        "UHS":    ["university of health sciences", "uhs"],
        "NU":     ["norton university"],
        "PUC":    ["pannasastra university"],
        "ZU":     ["zaman university"],
        "PIU":    ["paragon international university", "paragon"],
        "BBU":    ["build bright university"],
        "CUST":   ["cambodia university of technology and science"],
        "WU":     ["western university"],
        "IU":     ["international university"],
        "AU":     ["angkor university"],
        "NUBB":   ["national university of battambang"],
        "UC":     ["university of cambodia"],
        "LUCT":   ["limkokwing university"],
        "AUPP":   ["american university of phnom penh"],
        "CBS":    ["camed business school", "camed"],
        "CMU":    ["cambodian mekong university"],
        "SETEC":  ["setec institute"],
    }
    
    found_uni = False
    for short_code, variations in uni_map.items():
        for variation in variations:
            if variation in text_lower:
                data["School"] = short_code; found_uni = True; break
        if found_uni: break
    
    if not found_uni:
        school_match = re.search(r'([A-Z][A-Za-z\s&-]+(?:University|Institute|High\s*School))', text)
        if school_match: data["School"] = school_match.group(1).strip()

    # 7. EXPERIENCE
    exp_match = re.search(r'(?:WORK\s+)?(?:EXPERIENCE|EMPLOYMENT|HISTORY)[\s:.-]*(.*?)(?=\n(?:EDUCATION|SKILLS|REFERENCE|$))', text, re.IGNORECASE | re.DOTALL)
    if exp_match: 
        raw = re.sub(r'\s+', ' ', exp_match.group(1)).strip()
        data["Experience"] = raw[:350] + ("..." if len(raw) > 350 else "")

    # 8. GENDER
    if re.search(r'\b(?:Female|F)\b', text, re.IGNORECASE): data["Gender"] = "Female"
    elif re.search(r'\b(?:Male|M)\b', text, re.IGNORECASE): data["Gender"] = "Male"

    return data

# --- API ENDPOINTS ---

@app.post("/upload-cv")
async def upload_cv(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            content = await file.read()
            upload_result = cloudinary.uploader.upload(content, resource_type="auto", public_id=file.filename.split('.')[0])
            cv_url = upload_result.get("secure_url")
            raw_text = extract_text(content, file.filename)
            structured_data = parse_cv_text(raw_text)
            structured_data.update({"file_name": file.filename, "cv_url": cv_url, "upload_date": datetime.now().isoformat(), "locked": False})
            
            query = {"Name": structured_data["Name"], "Tel": structured_data["Tel"]}
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

# --- UPDATED CANDIDATES ENDPOINT (SEARCH + PAGINATION) ---
@app.get("/candidates")
async def get_candidates(
    page: int = Query(1, ge=1), 
    limit: int = Query(20, le=100),
    search: str = Query(None)
):
    # 1. Build Filter
    query_filter = {}
    if search:
        search_regex = {"$regex": search, "$options": "i"}
        query_filter = {
            "$or": [
                {"Name": search_regex},
                {"Tel": search_regex},
                {"School": search_regex},
                {"Location": search_regex}
            ]
        }

    # 2. Get Count
    total_count = await collection.count_documents(query_filter)

    # 3. Fetch Data
    skip = (page - 1) * limit
    cursor = collection.find(query_filter).sort("upload_date", -1).skip(skip).limit(limit)
    
    candidates = []
    async for candidate in cursor:
        candidate["_id"] = str(candidate["_id"])
        candidates.append(candidate)
        
    return {
        "data": candidates,
        "page": page,
        "limit": limit,
        "total": total_count
    }

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
        is_delete_all = len(request.candidate_ids) == 0
        if is_delete_all:
            if request.passcode != "9994": return {"status": "error", "message": "Invalid passcode"}
            await collection.delete_many({"locked": {"$ne": True}})
            return {"status": "success", "deleted": "All unlocked"}
        else:
            deleted_count = 0
            for cid in request.candidate_ids:
                obj_id = ObjectId(cid)
                candidate = await collection.find_one({"_id": obj_id})
                if candidate and not candidate.get("locked", False):
                    await collection.delete_one({"_id": obj_id})
                    deleted_count += 1
            return {"status": "success", "deleted": deleted_count}
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