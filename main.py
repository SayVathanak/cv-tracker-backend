from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
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
import phonenumbers 

# --- CONFIGURATION & MAPPINGS ---

# 1. UNIVERSITY MAPPING
UNI_MAP = {
    "RUPP":     ["royal university of phnom penh", "rupp"],
    "ITC":      ["institute of technology of cambodia", "itc", "techno"],
    "RULE":     ["royal university of law and economics", "rule"],
    "NUM":      ["national university of management", "num"],
    "RUA":      ["royal university of agriculture", "rua"],
    "AUPP":     ["american university of phnom penh", "aupp"],
    "PUC":      ["pannasastra university", "puc"],
    "BBU":      ["build bright university", "bbu"],
    "CUS":      ["cambodia university for specialties", "cus"],
    "AEU":      ["asia euro university", "aeu"],
    "HRU":      ["human resources university", "hru"],
    "CamTech":  ["camtech", "cambodia university of technology and science"],
    "AIB":      ["acleda institute of business", "aib"],
    "NPIC":     ["national polytechnic institute of cambodia", "npic"],
    "UC":       ["university of cambodia", "uc"],
    "Setec":    ["setec institute", "setec"],
    "Paragon":  ["paragon international university", "zaman"],
    "Norton":   ["norton university", "nu"],
    "Western":  ["western university", "wu"],
    "Beltei":   ["beltei international university", "beltei"],
    "Vanda":    ["vanda institute", "vanda"],
    "USEA":     ["university of south-east asia", "usea"],
    "NUBB":     ["national university of battambang", "nubb"],
    "MCU":      ["mean chey university", "mcu"],
    "IIC":      ["iic university of technology", "iic"],
    "Chenla":   ["chenla university"],
    "Kirirom":  ["kirirom institute", "kit"],
    "Limkokwing": ["limkokwing university"],
    "PPI":      ["phnom penh international university", "ppiu"]
}

# 2. PHNOM PENH LOCATIONS (Khan -> Sangkats)
# Logic: Try to match "Sangkat X, Khan Y"
PP_LOCATIONS = {
    "Khan Chamkar Mon": ["tonle bassac", "tuol tumpung", "boeung trabek", "psar daem thkov"],
    "Khan Daun Penh": ["phsar thmei", "chey chumneas", "srah chak", "wat phnom", "phsar kandal"],
    "Khan 7 Makara": ["monorom", "mittapheap", "veal vong", "orussey", "boeung prolit"],
    "Khan Toul Kork": ["boeung kak", "boeng kak", "phsar depo", "teuk l'ak", "teuk laak", "phsar daem kor"],
    "Khan Dangkao": ["dangkao", "pong tuek", "prey veng", "khmuonh"],
    "Khan Mean Chey": ["stung meanchey", "steung meanchey", "meanchey", "boeng tumpun", "chak angre"],
    "Khan Russey Keo": ["tuol sangke", "svay pak", "kilomaetr", "chrang chamreh"],
    "Khan Sen Sok": ["phnom penh thmei", "teuk thla", "khmuonh", "krang thnong"],
    "Khan Pou Senchey": ["kakab", "choam chao", "samraong krom"],
    "Khan Chroy Changvar": ["chroy changvar", "prek leap", "prek ta sek"],
    "Khan Chbar Ampov": ["chbar ampov", "prek pra", "nirouth", "kbal koh"],
    "Khan Boeung Keng Kang": ["boeung keng kang", "bkk1", "bkk2", "bkk3", "tuol svay prey"],
    "Khan Kamboul": ["kamboul", "kantan"]
}

# 3. SPECIAL PROVINCE CITIES (Format: "City, Province")
# If these cities/districts are found, we format as "City, Province"
SPECIAL_CITIES = {
    "Kandal": ["ta khmau", "takhmau", "ta khmao", "kien svay", "arey ksat", "lvea aem"],
    "Takeo": ["doun kaev", "daun keo", "tram kak", "bati"],
    "Kampong Speu": ["chbar mon"],
    "Siem Reap": ["siem reap municipality", "krong siem reap"],
    "Battambang": ["krong battambang"]
}

# 4. STANDARD PROVINCE MAPPING (Format: "Province Name")
PROVINCE_MAP = {
    "Siem Reap": ["siem reap"], 
    "Battambang": ["battambang"],
    "Kampong Cham": ["kampong cham"], 
    "Sihanoukville": ["preah sihanouk", "sihanoukville", "kompong som"],
    "Kandal": ["kandal"], 
    "Takeo": ["takeo"], 
    "Kampot": ["kampot"], 
    "Kep": ["kep"],
    "Koh Kong": ["koh kong"], 
    "Prey Veng": ["prey veng"], 
    "Svay Rieng": ["svay rieng", "bavet"],
    "Kampong Speu": ["kampong speu"], 
    "Kampong Thom": ["kampong thom"], 
    "Kampong Chhnang": ["kampong chhnang"],
    "Pursat": ["pursat"], 
    "Pailin": ["pailin"], 
    "Banteay Meanchey": ["banteay meanchey", "poipet"],
    "Oddar Meanchey": ["oddar meanchey"], 
    "Preah Vihear": ["preah vihear"], 
    "Stung Treng": ["stung treng"],
    "Ratanakiri": ["ratanakiri", "banlung"], 
    "Mondulkiri": ["mondulkiri", "sen monorom"], 
    "Kratie": ["kratie"], 
    "Tbong Khmum": ["tbong khmum", "tboung khmum"]
}

# --- SYSTEM CONFIG ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
else:
    print("Running on Linux/Cloud - using default Tesseract path")

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

@app.on_event("startup")
async def startup_db_client():
    await collection.create_index([("Name", "text"), ("Tel", "text"), ("School", "text"), ("Location", "text")])

# --- TEXT EXTRACTION ---

def _extract_text_sync(file_bytes: bytes, filename: str) -> str:
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
        print(f"Error reading file {filename}: {e}")
    return text

async def extract_text(file_bytes: bytes, filename: str) -> str:
    return await run_in_threadpool(_extract_text_sync, file_bytes, filename)

# --- PARSING LOGIC ---

def parse_cv_text(text: str) -> dict:
    data = {
        "Name": "N/A", "Birth": "N/A", "Tel": "N/A",
        "Location": "N/A", "School": "N/A", "Experience": "N/A",
        "Gender": "N/A"
    }
    
    # Normalize text
    text_normalized = re.sub(r'[\u200b\u200c\u200d\uFEFF]', '', text).strip()
    text_lower = text_normalized.lower()
    search_text = text_lower.replace(".", " ").replace(",", " ") # For cleaner matching
    
    # 1. NAME EXTRACTION
    best_name = ""
    text_for_name = text_normalized
    ref_match = re.search(r'\n(REFERENCE|REFERENCES|EXPERIENCE|WORK HISTORY)', text_normalized, re.IGNORECASE)
    if ref_match:
        text_for_name = text_normalized[:ref_match.start()]

    name_patterns = [
        r"(?:Name|Full\s*Name|ឈ្មោះ)[\s:.-]*([A-Z\u1780-\u17FF][a-zA-Z\u1780-\u17FF\s.]{2,50}?)(?=\n|Address|Date|Tel|Contact|Mobile|$)",
        r"^([A-Z\u1780-\u17FF][a-zA-Z\u1780-\u17FF\s.]{2,40})(?:\n|$)",
    ]
    
    for pattern in name_patterns:
        name_match = re.search(pattern, text_for_name, re.IGNORECASE | re.MULTILINE)
        if name_match:
            name = name_match.group(1).strip()
            exclude = ['resume', 'curriculum', 'contact', 'vitae', 'apply', 'summary', 'profile', 'personal']
            if len(name) > 3 and not any(w in name.lower() for w in exclude) and not re.search(r'\d', name):
                clean_name = re.sub(r'^(Ms\.|Mr\.|Mrs\.|Dr\.|លោក|អ្នកnang)\s*', '', name, flags=re.IGNORECASE)
                best_name = clean_name
                break
    
    if not best_name:
        first_line = text_normalized.split('\n')[0].strip()
        if 3 < len(first_line) < 50 and "resume" not in first_line.lower():
            best_name = re.sub(r'^(Ms\.|Mr\.|Mrs\.)\s*', '', first_line, flags=re.IGNORECASE)
    data["Name"] = best_name if best_name else "N/A"

    # 2. PHONE
    try:
        for match in phonenumbers.PhoneNumberMatcher(text_normalized, "KH"):
            data["Tel"] = phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.NATIONAL).replace(" ", "")
            break 
    except Exception: pass

    # 3. BIRTH DATE
    dob_match = re.search(
        r"(?:Birth|DOB|Born|Date of Birth)[\s:.-]*((?:\d{1,2}[-/\s\.]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/\s\.]+\d{4})|(?:\d{1,2}[-/\s\.]+\d{1,2}[-/\s\.]+\d{2,4}))", 
        text_normalized, re.IGNORECASE
    )
    if dob_match: data["Birth"] = dob_match.group(1).strip()

    # 4. LOCATION (Sangkat, Khan / City, Province)
    found_location = False
    
    # A. PHNOM PENH (Search for Sangkat AND Khan)
    for khan_name, sangkats in PP_LOCATIONS.items():
        # 1. Try to find Specific Sangkat
        sangkat_pattern = r'\b(?:sangkat|commune|s/k)?\s*(' + '|'.join(re.escape(s) for s in sangkats) + r')\b'
        sangkat_match = re.search(sangkat_pattern, search_text)
        
        if sangkat_match:
            # Result: "Sangkat X, Khan Y"
            clean_sangkat = sangkat_match.group(1).title()
            data["Location"] = f"Sangkat {clean_sangkat}, {khan_name}"
            found_location = True
            break
            
        # 2. If Sangkat not found, check Khan only
        clean_khan = khan_name.replace("Khan ", "").lower()
        khan_pattern = r'\b(?:khan|district|d\.)\s*' + re.escape(clean_khan) + r'\b'
        if re.search(khan_pattern, search_text):
            data["Location"] = khan_name
            found_location = True
            break
    
    # B. SPECIAL CITY CHECK (Kandal/Takeo/etc) - Format: "City, Province"
    if not found_location:
        for province, cities in SPECIAL_CITIES.items():
            city_pattern = r'\b(' + '|'.join(re.escape(c) for c in cities) + r')\b'
            city_match = re.search(city_pattern, search_text)
            if city_match:
                # Result: "Ta Khmau, Kandal"
                clean_city = city_match.group(1).title()
                data["Location"] = f"{clean_city}, {province}"
                found_location = True
                break

    # C. STANDARD PROVINCE CHECK - Format: "Province"
    if not found_location:
        for clean_prov, variations in PROVINCE_MAP.items():
            if clean_prov == "Phnom Penh": continue # Handle PP default last
            
            pattern = r'\b(' + '|'.join(re.escape(v) for v in variations) + r')\b'
            if re.search(pattern, search_text):
                data["Location"] = clean_prov
                found_location = True
                break
    
    # D. DEFAULT PHNOM PENH
    if not found_location:
        if "phnom penh" in search_text:
             data["Location"] = "Phnom Penh"
        else:
            # Fallback to whatever generic address line we found
            addr_match = re.search(r'(?:Address|Location|Addr|ទីលំនៅ)[\s:.-]*([^\n]{5,100})', text_normalized, re.IGNORECASE)
            if addr_match and len(addr_match.group(1)) < 50:
                 data["Location"] = addr_match.group(1).strip()

    # 5. SCHOOL
    found_uni = False
    for short_code, variations in UNI_MAP.items():
        for variation in variations:
            if variation in text_lower:
                data["School"] = short_code; found_uni = True; break
        if found_uni: break
    
    if not found_uni:
        school_match = re.search(r'([A-Z][A-Za-z\s&-]+(?:University|Institute|School))', text_normalized)
        if school_match: data["School"] = school_match.group(1).strip()

    # 6. EXPERIENCE
    exp_match = re.search(r'(?:WORK\s+)?(?:EXPERIENCE|EMPLOYMENT|HISTORY|Professional Background)[\s:.-]*(.*?)(?=\n(?:EDUCATION|SKILLS|REFERENCE|LANGUAGE|$))', text_normalized, re.IGNORECASE | re.DOTALL)
    if exp_match: 
        raw = re.sub(r'\s+', ' ', exp_match.group(1)).strip()
        data["Experience"] = raw[:500] + ("..." if len(raw) > 500 else "")

    # 7. GENDER
    if re.search(r'\b(?:Female|F)\b', text_normalized, re.IGNORECASE): data["Gender"] = "Female"
    elif re.search(r'\b(?:Male|M)\b', text_normalized, re.IGNORECASE): data["Gender"] = "Male"

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
            raw_text = await extract_text(content, file.filename)
            structured_data = parse_cv_text(raw_text)
            structured_data.update({
                "file_name": file.filename, 
                "cv_url": cv_url, 
                "upload_date": datetime.now().isoformat(), 
                "locked": False,
                "raw_text_snippet": raw_text[:200]
            })
            
            query = {"Name": structured_data["Name"], "Tel": structured_data["Tel"]}
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
async def get_candidates(
    page: int = Query(1, ge=1), 
    limit: int = Query(20, le=100),
    search: str = Query(None)
):
    query_filter = {}
    if search:
        search_regex = {"$regex": search, "$options": "i"}
        query_filter = {
            "$or": [
                {"Name": search_regex},
                {"Tel": search_regex},
                {"School": search_regex},
                {"Location": search_regex},
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
        if not candidate or "cv_url" not in candidate: 
            return Response(content="CV not found", status_code=404)
        
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
        is_delete_all = len(request.candidate_ids) == 0
        if is_delete_all:
            if request.passcode != "9994": 
                return {"status": "error", "message": "Invalid passcode"}
            await collection.delete_many({"locked": {"$ne": True}})
            return {"status": "success", "deleted": "All unlocked"}
        else:
            object_ids = [ObjectId(cid) for cid in request.candidate_ids]
            result = await collection.delete_many({
                "_id": {"$in": object_ids},
                "locked": {"$ne": True}
            })
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