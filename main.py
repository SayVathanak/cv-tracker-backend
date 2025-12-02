import os
import time
from typing import List, Optional
from datetime import datetime
import re
import tempfile
import asyncio
import json

# Web Framework
from fastapi import FastAPI, UploadFile, File, Query, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx 

# Database
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

# Cloud Services
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import google.generativeai as genai

load_dotenv()

# --- CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("No GEMINI_API_KEY found")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Cloudinary Config
cloudinary.config( 
  cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"), 
  api_key = os.getenv("CLOUDINARY_API_KEY"), 
  api_secret = os.getenv("CLOUDINARY_SECRET"),
  secure = True
)

# Database Config
MONGO_URL = os.getenv("MONGO_URL")
client = AsyncIOMotorClient(MONGO_URL)
db = client.cv_tracking_db
collection = db.candidates

# App Config
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- HELPER: BACKGROUND TASK LOGIC ---
async def process_cv_background(file_content: bytes, filename: str, cv_url: str, candidate_id: str, mime_type: str):
    """
    Runs in background: Uploads to Gemini -> Extracts Data -> Updates MongoDB
    """
    temp_path = None
    gemini_file = None
    
    try:
        print(f"[{filename}] Background task started...")

        # 1. Save bytes to a temp file (Gemini requirement)
        suffix = ".pdf" if mime_type == "application/pdf" else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            temp_path = tmp.name

        # 2. Upload to Gemini Files API
        gemini_file = genai.upload_file(path=temp_path, mime_type=mime_type)

        # 3. Wait for processing (Active State)
        while gemini_file.state.name == "PROCESSING":
            await asyncio.sleep(1)
            gemini_file = genai.get_file(gemini_file.name)

        if gemini_file.state.name == "FAILED":
            raise Exception("Gemini failed to process the file.")

        # 4. Generate Content (Gemini 2.5 Flash)
        model = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
        
        # --- PROMPT WITH EXCEL-FRIENDLY DATE FORMAT ---
        prompt = """
        You are an expert HR Data Extractor for Cambodian Candidates.
        Analyze the uploaded CV and extract the details into a JSON object.

        ### RULES FOR LOCATION (Cambodia Context):
        1. **Phnom Penh:** Format as "Sangkat [Name], Khan [Name]" or "Khan [Name]". 
        2. **Provinces:** Return "City, Province" or just "Province Name".

        ### RULES FOR SCHOOL (Education):
        - Extract the HIGHEST education level.
        - Priority 1: University/Institute name (e.g., RUPP, SETEC, NUM).
        - Priority 2: If no university found, extract High School name.
        - **Format:** School Name ONLY. Do NOT include degree (Bachelor/Master), dates, or GPA.

        ### RULES FOR POSITION:
        - Extract the Job Title the candidate is applying for.
        - Look for "Applying for...", "Subject: Application for...", "Objective", or a professional title under their name.
        - If not mentioned, return "N/A".

        ### RULES FOR OTHER FIELDS:
        - **Name:** Full name (Capitalize properly). Remove titles.
        - **Tel:** Standard local format (e.g., "012 345 678"). REMOVE country codes like +855.
        - **Experience:** Summarize last job title and company (< 15 words).
        - **Gender:** Detect Male/Female.
        
        ### RULE FOR BIRTHDATE (Important for Excel):
        - **Format:** Strictly 'DD-Mon-YYYY' (e.g., '11-Dec-2002' or '01-Jan-1999').
        - Convert numeric months to English 3-letter abbreviations (Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec).
        - If only Year is found, return "01-Jan-YYYY".
        - If no date found, return "N/A".

        Return JSON with these exact keys:
        {
            "Name": "Full Name",
            "Tel": "0xx xxx xxx",
            "Location": "Sangkat, Khan (or Province)",
            "School": "School Name ONLY",
            "Experience": "Last Job Title & Company",
            "Gender": "Male/Female/N/A",
            "BirthDate": "DD-Mon-YYYY",
            "Position": "Role they are applying for (or N/A)"
        }
        """
        
        response = model.generate_content([gemini_file, prompt])
        
        # 5. Clean Data
        try:
            # Sometimes Gemini adds markdown code blocks, strip them
            json_text = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(json_text)
        except:
            data = {"Name": "Parse Error", "Experience": "AI returned invalid JSON"}

        # 6. Update MongoDB with Real Data
        update_payload = {
            "Name": data.get("Name", "N/A"),
            "Tel": data.get("Tel", "N/A"),
            "Location": data.get("Location", "N/A"),
            "School": data.get("School", "N/A"),
            "Experience": data.get("Experience", "N/A"),
            "Gender": data.get("Gender", "N/A"),
            "BirthDate": data.get("BirthDate", "N/A"),
            "Position": data.get("Position", "N/A"),
            "status": "Ready",  # MARK AS DONE
            "last_modified": datetime.now().isoformat()
        }

        await collection.update_one(
            {"_id": ObjectId(candidate_id)}, 
            {"$set": update_payload}
        )
        print(f"[{filename}] Success! DB Updated.")

    except Exception as e:
        print(f"[{filename}] Error: {e}")
        await collection.update_one(
            {"_id": ObjectId(candidate_id)}, 
            {"$set": {"status": "Error", "error_msg": str(e)}}
        )

    finally:
        # Cleanup: Delete the temp file and the file on Gemini's server
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if gemini_file:
            try:
                genai.delete_file(gemini_file.name)
            except:
                pass

# --- API ENDPOINTS ---

@app.post("/upload-cv")
async def upload_cv(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    results = []
    
    for file in files:
        try:
            # 1. Read file into memory
            content = await file.read()
            mime_type = file.content_type or "application/pdf"
            
            # 2. Upload to Cloudinary (Sync)
            clean_name = re.sub(r'[^a-zA-Z0-9]', '_', file.filename.split('.')[0])
            upload_result = cloudinary.uploader.upload(content, resource_type="auto", public_id=clean_name)
            cv_url = upload_result.get("secure_url")
            
            # 3. Create "Placeholder" Candidate
            placeholder_data = {
                "Name": "Processing...", 
                "Tel": "...", 
                "Location": "...", 
                "School": "...", 
                "Experience": "AI is analyzing...", 
                "Gender": "...",
                "BirthDate": "...",
                "Position": "...",
                "file_name": file.filename,
                "cv_url": cv_url,
                "upload_date": datetime.now().isoformat(),
                "locked": False,
                "status": "Processing"
            }
            
            insert_result = await collection.insert_one(placeholder_data)
            candidate_id = str(insert_result.inserted_id)
            
            # 4. Trigger Background Task
            background_tasks.add_task(
                process_cv_background, 
                content, 
                file.filename, 
                cv_url, 
                candidate_id,
                mime_type
            )
            
            placeholder_data["_id"] = candidate_id
            results.append(placeholder_data)

        except Exception as e:
            print(f"Upload Error {file.filename}: {e}")
            results.append({"filename": file.filename, "status": "Error", "details": str(e)})
            
    return {"status": f"Queued {len(results)} files", "details": results}

@app.get("/candidates")
async def get_candidates(page: int = Query(1, ge=1), limit: int = Query(20, le=100), search: str = Query(None)):
    query_filter = {}
    if search:
        search_regex = {"$regex": search, "$options": "i"}
        query_filter = {
            "$or": [
                {"Name": search_regex}, {"Tel": search_regex},
                {"School": search_regex}, {"Location": search_regex}
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
        
        if not candidate or "cv_url" not in candidate:
            return Response(content="CV URL missing in database", status_code=404)
            
        # Use httpx to stream the file from Cloudinary
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(candidate["cv_url"])
            if response.status_code != 200:
                 return Response(content=f"Cloudinary Error: {response.status_code}", status_code=404)
        
        media_type = "application/pdf"
        url_lower = candidate["cv_url"].lower()
        if any(ext in url_lower for ext in [".jpg", ".jpeg", ".png"]):
            media_type = "image/jpeg"

        return Response(content=response.content, media_type=media_type)
        
    except Exception as e:
        print(f"CV Download Error: {e}")
        return Response(content=f"Server Error: {str(e)}", status_code=500)

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

# --- BULK DELETE & RETRY ENDPOINTS ---

class BulkDeleteRequest(BaseModel):
    candidate_ids: List[str] = []
    passcode: str = None

@app.post("/candidates/bulk-delete")
async def bulk_delete_candidates(request: BulkDeleteRequest):
    try:
        SERVER_PASSCODE = os.getenv("ADMIN_PASSCODE", "9994")

        # SCENARIO 1: DELETE ALL
        if not request.candidate_ids and request.passcode:
            if request.passcode != SERVER_PASSCODE:
                return {"status": "error", "message": "Wrong Admin Passcode"}
            result = await collection.delete_many({"locked": {"$ne": True}})
            return {"status": "success", "message": f"Wiped {result.deleted_count} records"}

        # SCENARIO 2: DELETE SELECTED
        elif request.candidate_ids:
            ids_to_delete = []
            for cid in request.candidate_ids:
                try:
                    ids_to_delete.append(ObjectId(cid))
                except:
                    continue
            result = await collection.delete_many({
                "_id": {"$in": ids_to_delete},
                "locked": {"$ne": True}
            })
            return {"status": "success", "message": f"Deleted {result.deleted_count} candidates"}

        return {"status": "error", "message": "Invalid request"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/candidates/{candidate_id}/retry")
async def retry_parsing(candidate_id: str, background_tasks: BackgroundTasks):
    try:
        candidate = await collection.find_one({"_id": ObjectId(candidate_id)})
        if not candidate:
            return {"status": "error", "message": "Candidate not found"}

        async with httpx.AsyncClient() as client:
            response = await client.get(candidate["cv_url"])
            if response.status_code != 200:
                return {"status": "error", "message": "Failed to download CV"}
            file_content = response.content

        mime_type = "application/pdf"
        if any(ext in candidate["cv_url"].lower() for ext in [".jpg", ".jpeg", ".png"]):
            mime_type = "image/jpeg"

        await collection.update_one(
            {"_id": ObjectId(candidate_id)},
            {"$set": {"status": "Processing", "Name": "Retrying..."}}
        )

        background_tasks.add_task(
            process_cv_background, 
            file_content, 
            candidate.get("file_name", "retry"), 
            candidate["cv_url"], 
            candidate_id, 
            mime_type
        )
        return {"status": "success", "data": {"status": "Processing"}}
    except Exception as e:
        return {"status": "error", "message": str(e)}