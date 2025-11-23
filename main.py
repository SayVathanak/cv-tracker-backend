from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
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

# --- CLOUDINARY SETUP (REPLACE WITH YOUR KEYS) ---
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
    data = {
        "Name": "N/A", "Birth": "N/A", "Tel": "N/A", "Email": "N/A",
        "Location": "N/A", "School": "N/A", "Experience": "N/A",
        "Skills": "N/A", "Education_Level": "N/A"
    }
    
    text_normalized = re.sub(r'\s+', ' ', text)
    
    # 1. NAME
    name_patterns = [
        r"(?:Full\s*)?Name[\s:.-]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
        r"(?:ឈ្មោះ|Name)[\s:.-]*([^\n\d]+?)(?:\n|Address|Date)",
    ]
    for pattern in name_patterns:
        name_match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if name_match:
            name = name_match.group(1).strip()
            if len(name) > 3 and len(name) < 50 and not any(w in name.lower() for w in ['resume', 'curriculum']):
                data["Name"] = name
                break
    if data["Name"] == "N/A":
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines and "resume" not in lines[0].lower(): data["Name"] = lines[0]

    # 2. EMAIL
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match: data["Email"] = email_match.group(0).strip()

    # 3. PHONE
    phone_patterns = [r'\+855[\s-]?\d{1,2}[\s-]?\d{3}[\s-]?\d{3,4}', r'0\d{1,2}[\s-]?\d{3}[\s-]?\d{3,4}']
    for pattern in phone_patterns:
        phone_match = re.search(pattern, text)
        if phone_match and len(re.sub(r'\D', '', phone_match.group(0))) >= 8:
            data["Tel"] = phone_match.group(0).strip(); break

    # 4. BIRTH DATE
    # Updated regex to handle "June 08th, 2004"
    dob_match = re.search(r"(?:Birth|DOB).*?(\d{1,2}[-/thstndrd\s]+[A-Za-z0-9]+[-/,\s]+\d{4}|[A-Za-z]+\s+\d{1,2}[thstndrd,]*\s+\d{4})", text, re.IGNORECASE | re.DOTALL)
    if dob_match: 
        data["Birth"] = dob_match.group(1).replace('\n', ' ').strip()

    # 5. LOCATION
    loc_match = re.search(r'(?:Address|Location)[\s:.-]*(.*?(?:Province|City|Street|Phnom\s*Penh))', text, re.IGNORECASE | re.DOTALL)
    if loc_match: data["Location"] = loc_match.group(1).replace("\n", " ").strip()

    # 6. EDUCATION
    school_match = re.search(r'([A-Za-z\s&]+(?:University|Institute|College|High\s*School))', text, re.IGNORECASE)
    if school_match: data["School"] = re.sub(r'\d+', '', school_match.group(1).strip())

    # 7. EXPERIENCE
    exp_match = re.search(r'(?:Work\s*Experience|History)(.*?)(?=\n(?:Education|Skills|Reference|$))', text, re.IGNORECASE | re.DOTALL)
    if exp_match:
        raw_exp = exp_match.group(1).strip()
        if len(raw_exp) > 10: data["Experience"] = raw_exp[:200] + "..."

    # 8. SKILLS
    skills_match = re.search(r'(?:Skills)(.*?)(?=\n(?:Experience|Education|$))', text, re.IGNORECASE | re.DOTALL)
    if skills_match:
        data["Skills"] = skills_match.group(1).strip()[:100]

    return data

# --- API ENDPOINTS ---

@app.post("/upload-cv")
async def upload_cv(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            # 1. Read content for OCR
            content = await file.read()
            
            # 2. UPLOAD TO CLOUDINARY
            # This sends the file to Cloudinary and gets a permanent URL back
            upload_result = cloudinary.uploader.upload(content, resource_type="auto", public_id=file.filename.split('.')[0])
            cv_url = upload_result.get("secure_url")

            # 3. Extract & Parse
            raw_text = extract_text(content, file.filename)
            structured_data = parse_cv_text(raw_text)
            
            structured_data["file_name"] = file.filename
            structured_data["cv_url"] = cv_url  # <--- SAVE THE URL, NOT THE FILE
            structured_data["upload_date"] = datetime.now().isoformat()
            
            # 4. Database Upsert
            query = {"Name": structured_data["Name"], "Tel": structured_data["Tel"]}
            if structured_data["Email"] != "N/A":
                query = {"$or": [query, {"Email": structured_data["Email"]}]}

            existing = await collection.find_one(query)
            
            if existing:
                await collection.update_one({"_id": existing["_id"]}, {"$set": structured_data})
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
        
        # Use httpx to fetch the file from the external Cloudinary URL
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(cv_url)
            response.raise_for_status() # Raise an exception for 4xx or 5xx status codes

        # Return the file content with the correct MIME type
        content_type = response.headers.get("Content-Type", "application/pdf")
        
        # FastAPI's CORS middleware will automatically add the necessary 
        # Access-Control-Allow-Origin: * header to this response.
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
        await collection.delete_one({"_id": obj_id})
        return {"status": "Deleted successfully"}
    except Exception as e:
        return {"status": f"Error: {e}"}

@app.put("/candidates/{candidate_id}")
async def update_candidate(candidate_id: str, updated_data: dict):
    try:
        obj_id = ObjectId(candidate_id)
        if "_id" in updated_data: del updated_data["_id"]
        updated_data["last_modified"] = datetime.now().isoformat()
        await collection.update_one({"_id": obj_id}, {"$set": updated_data})
        return {"status": "Updated successfully"}
    except Exception as e:
        return {"status": f"Error: {e}"}
    
