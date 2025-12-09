import os
import time
from typing import List, Optional
from datetime import datetime, timedelta, timezone  
import re
import tempfile
import asyncio
import json
import logging
import traceback
import uuid

# Web Framework
from fastapi import FastAPI, UploadFile, File, Query, BackgroundTasks, Response, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import httpx 
from apscheduler.schedulers.background import BackgroundScheduler

# Security
from passlib.context import CryptContext
from jose import JWTError, jwt
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from bakong_khqr import KHQR

# Database
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

# Cloud Services
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import google.generativeai as genai

# --- 1. LOGGING & CONFIGURATION ---
load_dotenv()

# Configure Logging (Prints timestamps and error details to console)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Validate API Keys
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("No GEMINI_API_KEY found. Application may not function correctly.")

genai.configure(api_key=GEMINI_API_KEY)

cloudinary.config( 
  cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"), 
  api_key = os.getenv("CLOUDINARY_API_KEY"), 
  api_secret = os.getenv("CLOUDINARY_SECRET"),
  secure = True
)

# Database Setup
MONGO_URL = os.getenv("MONGO_URL")
client = AsyncIOMotorClient(MONGO_URL)
db = client.cv_tracking_db
collection = db.candidates
users_collection = db["users"]
transactions_collection = db["transactions"]

# Payment Setup
BAKONG_DEV_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRhIjp7ImlkIjoiZmU2MzNjNDdlOGZlNDQ0YSJ9LCJpYXQiOjE3NjQ5NTIyNDksImV4cCI6MTc3MjcyODI0OX0.tbhgtVlzNrTGhD0mKkN33BgopmENupueM7qa9DsDxOI"
BAKONG_ACCOUNT_ID = "say_vathanak@aclb"
MERCHANT_NAME = "SAY SAKSOVATHANAK"
MERCHANT_CITY = "Phnom Penh"
khqr = KHQR(BAKONG_DEV_TOKEN) 

# Auth Setup
SECRET_KEY = os.getenv("SECRET_KEY", "YOUR_SUPER_SECRET_KEY_CHANGE_THIS_IN_PROD")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
pwd_context = CryptContext(schemes=["bcrypt"], deprecated=["auto"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- 2. BACKGROUND TASKS ---
async def run_async_cleanup():
    """
    Deletes files ONLY for users who have enabled autoDelete in their settings.
    Respects the specific retention period set by the user.
    """
    logger.info("[Auto-Cleanup] Scanning for files to delete...")
    
    # 1. Find all users who have enabled autoDelete
    # Note: We look for users where settings.autoDelete is specifically true
    async for user in users_collection.find({"settings.autoDelete": True}):
        try:
            username = user.get("username")
            settings = user.get("settings", {})
            
            # Get user's specific retention (default to 30 days if invalid)
            try:
                days = int(settings.get("retention", "30"))
            except:
                days = 30
                
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # 2. Find files uploaded by THIS user that are older than the cutoff
            cursor = collection.find({
                "uploaded_by": username,
                "upload_date": {"$lt": cutoff_time.isoformat()},
                "cv_url": {"$ne": None}, 
                "file_status": {"$ne": "Expired"},
                "locked": {"$ne": True} # Never delete locked files
            })

            async for candidate in cursor:
                url = candidate.get("cv_url")
                if not url or "http" not in url: continue 

                try:
                    public_id = url.split('/')[-1].split('.')[0]
                    logger.info(f" -> Deleting PDF for user {username}: {candidate.get('Name')}")
                    cloudinary.uploader.destroy(public_id)
                    
                    await collection.update_one(
                        {"_id": candidate["_id"]},
                        {"$set": {"cv_url": None, "file_status": "Expired"}}
                    )
                except Exception as e:
                    logger.error(f"Error cleaning {candidate.get('_id')}: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing user {user.get('username')}: {e}")

async def process_cv_background(file_content: bytes, filename: str, cv_url: str, candidate_id: str, mime_type: str):
    """
    Background Task: 
    1. Uploads file to Gemini
    2. Extracts Data using AI
    3. Updates MongoDB with results or error message
    """
    temp_path = None
    gemini_file = None
    
    try:
        logger.info(f"[{filename}] 🚀 Started AI processing...")

        suffix = ".pdf" if mime_type == "application/pdf" else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            temp_path = tmp.name

        gemini_file = genai.upload_file(path=temp_path, mime_type=mime_type)

        # Wait for Gemini processing
        while gemini_file.state.name == "PROCESSING":
            await asyncio.sleep(1)
            gemini_file = genai.get_file(gemini_file.name)

        if gemini_file.state.name == "FAILED":
            raise Exception("Gemini failed to process the file media.")

        model = genai.GenerativeModel('gemini-2.0-flash', generation_config={"response_mime_type": "application/json"})
        
        prompt = """
        You are an expert HR Data Extractor for candidates in Cambodia.
        Analyze the uploaded CV (PDF or Image) and extract details into a strict JSON object.

        ### 1. EXTRACTION RULES:
        - **Name:** Full Name in Title Case (e.g., "Sokha Dara"). If name is in Khmer, Romanize it or keep it if standard.
        - **Tel:** Standardize to "0xx xxx xxx" (e.g., 012 999 888). Remove +855 prefix if present.
        - **Location:** Extract "City" or "District, City". If "Phnom Penh", just return "Phnom Penh".
        - **School:** Extract the most recent University/Institute name only. Use acronyms if common (e.g., "RUPP", "ITC", "Setec").
        - **EducationLevel:** Choose exactly one: ['High School', 'Associate', 'Bachelor', 'Master', 'PhD', 'Other'].
        - **Gender:** ['Male', 'Female', 'N/A']. Infer from photo or name if not explicitly stated.
        
        ### 2. EXPERIENCE SUMMARY (Crucial):
        - Format: "Role at Company". 
        - Example: "Sales Manager at ABC Co".
        - If multiple jobs, take the MOST RECENT one.
        - Max 10 words. Keep it short for a dashboard card.
        - If Fresh Graduate, return "Fresh Graduate".

        ### 3. CONFIDENCE SCORE (0-100):
        - 100 = Clear text, all fields found.
        - 80 = Missing 1 minor field (e.g., Address).
        - 50 = Scanned image, blurry, or missing key info like Tel/Name.

        Return strictly this JSON structure:
        {
            "Name": "String",
            "Tel": "String",
            "Location": "String",
            "School": "String",
            "EducationLevel": "String",
            "Experience": "String",
            "Gender": "String",
            "BirthDate": "DD-Mon-YYYY",
            "Position": "String",
            "Confidence": Integer
        }
        """
        
        response = await model.generate_content_async([gemini_file, prompt])
        
        try:
            json_text = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(json_text)
            if isinstance(data, list): data = data[0] if len(data) > 0 else {}
        except Exception as parse_error:
            logger.warning(f"[{filename}] JSON Parse Error: {parse_error}")
            data = {"Name": "Parse Error", "Confidence": 0, "Experience": "AI output invalid JSON"}

        update_payload = {
            "Name": data.get("Name", "N/A"),
            "Tel": data.get("Tel", "N/A"),
            "Location": data.get("Location", "N/A"),
            "School": data.get("School", "N/A"),
            "EducationLevel": data.get("EducationLevel", "Other"), 
            "Experience": data.get("Experience", "N/A"),
            "Gender": data.get("Gender", "N/A"),
            "BirthDate": data.get("BirthDate", "N/A"),
            "Position": data.get("Position", "N/A"),
            "Confidence": data.get("Confidence", 0), 
            "status": "Ready",
            "last_modified": datetime.now().isoformat()
        }

        await collection.update_one({"_id": ObjectId(candidate_id)}, {"$set": update_payload})
        logger.info(f"[{filename}] ✅ Success! Confidence: {data.get('Confidence')}")

    except Exception as e:
        logger.error(f"[{filename}] ❌ Background Processing Failed: {str(e)}")
        logger.error(traceback.format_exc())

        # Friendly error message for the user DB record
        user_msg = "Failed to analyze CV."
        if "429" in str(e): user_msg = "System is busy (AI Quota). Retry later."
        elif "json" in str(e).lower(): user_msg = "AI could not read document format."
        
        await collection.update_one(
            {"_id": ObjectId(candidate_id)}, 
            {"$set": {"status": "Error", "error_msg": user_msg}}
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except: pass
        if gemini_file:
            try: genai.delete_file(gemini_file.name)
            except: pass

# --- 3. APP INITIALIZATION & GLOBAL HANDLERS ---
app = FastAPI()

# Handler 1: Catch Known HTTP Exceptions (like 402 Insufficient Credits)
# This allows the specific "detail" message to reach the frontend.
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )

# Handler 2: Catch Unexpected Crashes (500)
# Logs the error for you, gives a generic message to the user.
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_id = uuid.uuid4()
    logger.error(f"--- Unhandled Error ID: {error_id} | Path: {request.url.path} ---")
    logger.error(f"Details: {str(exc)}")
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An unexpected system error occurred. Please contact support.",
            "error_id": str(error_id)
        },
    )

ALLOWED_ORIGINS = ["http://localhost:5173", "https://cvtracker-kh.vercel.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. DATA MODELS ---
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class BulkDeleteRequest(BaseModel):
    candidate_ids: List[str] = []
    
class GoogleAuthRequest(BaseModel):
    token: str
    
class PaymentRequest(BaseModel):
    package_id: str
    email: str
    
class UserSettings(BaseModel):
    autoDelete: bool = False
    retention: str = "30"
    exportFields: dict = {}
    autoTags: str = ""
    profile: dict = {}

# --- 5. HELPER FUNCTIONS ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None: raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
        
    user = await users_collection.find_one({"username": username})
    if user is None: raise HTTPException(status_code=401, detail="User not found")
    return user

async def add_credits(user_id, amount: int, reason: str, ref: str = None):
    await transactions_collection.insert_one({
        "user_id": user_id,
        "amount": amount,
        "type": "PURCHASE" if amount > 0 else "SPEND",
        "description": reason,
        "payment_ref": ref,
        "created_at": datetime.now().isoformat(),
        "status": "COMPLETED"
    })
    await users_collection.update_one({"_id": user_id}, {"$inc": {"current_credits": amount}})

# --- 6. API ENDPOINTS ---

@app.on_event("startup")
async def start_scheduler():
    logger.info("--> System Startup: Checking for expired files...")
    await run_async_cleanup()

@app.post("/register", status_code=201)
async def register(user: UserCreate):
    if await users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    new_user = {
        "username": user.username,
        "hashed_password": get_password_hash(user.password),
        "current_credits": 0,
        "created_at": datetime.now().isoformat(),
        "provider": "local"
    }
    result = await users_collection.insert_one(new_user)
    await add_credits(result.inserted_id, 10, "Welcome Gift: Free 10 Credits", "SIGNUP_BONUS")
    return {"message": "User created successfully"}

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await users_collection.find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    token = create_access_token(data={"sub": user["username"]}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer"}

@app.post("/auth/google")
async def google_login(request: GoogleAuthRequest):
    try:
        id_info = id_token.verify_oauth2_token(request.token, google_requests.Request(), GOOGLE_CLIENT_ID)
        email = id_info.get("email")
        
        user = await users_collection.find_one({"username": email})
        if not user:
            result = await users_collection.insert_one({
                "username": email, "hashed_password": "GOOGLE_AUTH_USER",
                "provider": "google", "created_at": datetime.now().isoformat(), "current_credits": 0
            })
            await add_credits(result.inserted_id, 10, "Welcome Gift", "SIGNUP_BONUS")
        
        token = create_access_token(data={"sub": email}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        return {"access_token": token, "token_type": "bearer", "username": email}
    except Exception as e:
        logger.error(f"Google Auth Error: {e}")
        raise HTTPException(status_code=400, detail="Google Authentication failed")

@app.get("/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return {
        "username": current_user["username"],
        "current_credits": current_user.get("current_credits", 0),
        "settings": current_user.get("settings", {}) # Return saved settings
    }

@app.post("/upload-cv")
async def upload_cv(
    background_tasks: BackgroundTasks, 
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Handles file upload with credit deduction.
    If credits are insufficient for the batch, it raises a 402 Error.
    """
    results = []
    cost = len(files)
    user_credits = current_user.get("current_credits", 0)
    
    if user_credits < cost:
        raise HTTPException(
            status_code=402,
            # This text is what the user sees in the pop-up
            detail=f"Your credit balance is too low to upload {cost} files. Please top up your account to continue."
        )

    # Deduct credits
    await users_collection.update_one({"_id": current_user["_id"]}, {"$inc": {"current_credits": -cost}})
    
    await transactions_collection.insert_one({
        "user_id": current_user["_id"], "amount": -cost, "type": "SPEND",
        "description": f"Uploaded {cost} CV(s)", "created_at": datetime.now().isoformat(), "status": "COMPLETED"
    })

    for file in files:
        try:
            content = await file.read()
            clean_name = re.sub(r'[^a-zA-Z0-9]', '_', file.filename.split('.')[0])
            
            try:
                upload_result = cloudinary.uploader.upload(content, resource_type="auto", public_id=clean_name)
            except Exception as e:
                logger.error(f"Cloudinary Error: {e}")
                raise Exception("File storage service is unavailable.")

            cv_url = upload_result.get("secure_url")
            
            placeholder_data = {
                "Name": "Processing...", "Tel": "...", "Location": "...", "School": "...", 
                "Experience": "AI is analyzing...", "Gender": "...", "BirthDate": "...", "Position": "...",
                "file_name": file.filename, "cv_url": cv_url, "upload_date": datetime.now().isoformat(),
                "locked": False, "status": "Processing", "uploaded_by": current_user["username"]
            }
            
            insert_result = await collection.insert_one(placeholder_data)
            candidate_id = str(insert_result.inserted_id)
            
            background_tasks.add_task(
                process_cv_background, content, file.filename, cv_url, candidate_id, file.content_type
            )
            
            placeholder_data["_id"] = candidate_id
            results.append(placeholder_data)

        except Exception as e:
            logger.error(f"Upload Error {file.filename}: {e}", exc_info=True)
            results.append({"filename": file.filename, "status": "Error", "details": str(e)})
            
    return {"status": f"Queued {len(results)} files", "details": results, "remaining_credits": user_credits - cost}

@app.get("/candidates")
async def get_candidates(
    page: int = Query(1, ge=1), 
    limit: int = Query(20, le=100), 
    search: str = Query(None),
    current_user: dict = Depends(get_current_user)
):
    query_filter = {"uploaded_by": current_user["username"]}
    if search:
        search_regex = {"$regex": search, "$options": "i"}
        # --- FIX: Added {"Position": search_regex} to the list below ---
        query_filter["$and"] = [{
            "$or": [
                {"Name": search_regex}, 
                {"Tel": search_regex}, 
                {"School": search_regex}, 
                {"Location": search_regex},
                {"Position": search_regex} 
            ]
        }]

    total_count = await collection.count_documents(query_filter)
    cursor = collection.find(query_filter).sort("upload_date", -1).skip((page - 1) * limit).limit(limit)
    candidates = [ {**c, "_id": str(c["_id"])} async for c in cursor ]
        
    return {"data": candidates, "page": page, "limit": limit, "total": total_count}

@app.get("/cv/{candidate_id}")
async def get_candidate_cv(candidate_id: str):
    try:
        candidate = await collection.find_one({"_id": ObjectId(candidate_id)})
        if not candidate: return Response(content="Not Found", status_code=404)
            
        async with httpx.AsyncClient() as client:
            response = await client.get(candidate["cv_url"])
        
        media_type = "image/jpeg" if any(x in candidate["cv_url"] for x in [".jpg", ".png"]) else "application/pdf"
        return Response(content=response.content, media_type=media_type)
    except Exception as e:
        logger.error(f"Download Error: {e}")
        return Response(content="Server Error", status_code=500)

@app.delete("/candidates/{candidate_id}")
async def delete_candidate(candidate_id: str, current_user: dict = Depends(get_current_user)):
    candidate = await collection.find_one({"_id": ObjectId(candidate_id), "uploaded_by": current_user["username"]})
    if not candidate: raise HTTPException(status_code=404, detail="Candidate not found")
    if candidate.get("locked"): raise HTTPException(status_code=403, detail="Candidate is locked")
    
    await collection.delete_one({"_id": ObjectId(candidate_id)})
    return {"status": "Deleted successfully"}

@app.put("/candidates/{candidate_id}")
async def update_candidate(candidate_id: str, updated_data: dict, current_user: dict = Depends(get_current_user)):
    if "_id" in updated_data: del updated_data["_id"]
    updated_data["last_modified"] = datetime.now().isoformat()
    
    res = await collection.update_one(
        {"_id": ObjectId(candidate_id), "uploaded_by": current_user["username"]}, 
        {"$set": updated_data}
    )
    if res.matched_count == 0: raise HTTPException(status_code=404, detail="Candidate not found")
    return {"status": "Updated successfully"}

@app.put("/candidates/{candidate_id}/lock")
async def toggle_lock(candidate_id: str, request: dict, current_user: dict = Depends(get_current_user)):
    res = await collection.update_one(
        {"_id": ObjectId(candidate_id), "uploaded_by": current_user["username"]}, 
        {"$set": {"locked": request.get("locked", False)}}
    )
    if res.matched_count == 0: raise HTTPException(status_code=404, detail="Candidate not found")
    return {"status": "success"}

@app.put("/users/settings")
async def update_user_settings(settings: UserSettings, current_user: dict = Depends(get_current_user)):
    settings_dict = settings.dict()
    await users_collection.update_one(
        {"_id": current_user["_id"]},
        {"$set": {"settings": settings_dict}}
    )
    return {"status": "success", "settings": settings_dict}

@app.post("/candidates/bulk-delete")
async def bulk_delete_candidates(request: BulkDeleteRequest, current_user: dict = Depends(get_current_user)):
    user_filter = {"uploaded_by": current_user["username"], "locked": {"$ne": True}}
    
    if request.candidate_ids:
        user_filter["_id"] = {"$in": [ObjectId(cid) for cid in request.candidate_ids]}
        
    result = await collection.delete_many(user_filter)
    return {"status": "success", "message": f"Deleted {result.deleted_count} candidates"}
    
@app.post("/candidates/{candidate_id}/retry")
async def retry_parsing(candidate_id: str, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user)):
    candidate = await collection.find_one({"_id": ObjectId(candidate_id), "uploaded_by": current_user["username"]})
    if not candidate: raise HTTPException(status_code=404, detail="Candidate not found")

    async with httpx.AsyncClient() as client:
        response = await client.get(candidate["cv_url"])
    
    background_tasks.add_task(
        process_cv_background, response.content, candidate.get("file_name", "retry"), 
        candidate["cv_url"], candidate_id, "application/pdf"
    )
    await collection.update_one({"_id": ObjectId(candidate_id)}, {"$set": {"status": "Processing"}})
    return {"status": "success"}
    
@app.post("/api/create-payment")
async def create_khqr_payment(request: PaymentRequest):
    packages = {"small": {"price": 1.00, "credits": 20}, "pro": {"price": 5.00, "credits": 150}}
    if request.package_id not in packages: raise HTTPException(status_code=400, detail="Invalid package")
    
    pkg = packages[request.package_id]
    bill_number = str(uuid.uuid4().int)[:10] 

    try:
        qr = khqr.create_qr(
            bank_account=BAKONG_ACCOUNT_ID, merchant_name=MERCHANT_NAME, merchant_city=MERCHANT_CITY,
            amount=pkg["price"], currency="USD", phone_number='85592886006',
            store_label="CV Credits", bill_number=bill_number, terminal_label="POS-01"
        )
    except Exception as e:
        logger.error(f"Bakong Error: {e}")
        raise HTTPException(status_code=503, detail="Payment service unavailable")
    
    md5_hash = khqr.generate_md5(qr)
    user = await users_collection.find_one({"username": request.email})
    if user:
        await transactions_collection.insert_one({
            "user_id": user["_id"], "amount": pkg["credits"], "type": "PURCHASE_INTENT",
            "status": "PENDING", "payment_ref": bill_number, "md5_hash": md5_hash, "created_at": datetime.now().isoformat()
        })

    return {"qr_code": qr, "md5": md5_hash, "amount": pkg["price"]}
    
# Search for this endpoint in main.py and REPLACE it with this version:

@app.post("/api/check-payment-status")
async def check_payment_status(md5_hash: str, force: bool = Query(False)):
    """
    Checks payment status. 
    Added 'force=True' to simulate payment success in Dev mode.
    """
    # 1. Find the transaction
    trx = await transactions_collection.find_one({"md5_hash": md5_hash})
    
    if not trx: 
        raise HTTPException(status_code=404, detail="Transaction not found")
        
    # 2. If already paid, return success immediately
    if trx.get("status") == "COMPLETED": 
        user = await users_collection.find_one({"_id": trx["user_id"]})
        return {
            "status": "PAID", 
            "message": "Already processed", 
            "new_credits": trx["amount"],
            "total_credits": user.get("current_credits", 0)
        }

    # 3. Check Real Status
    payment_status = "UNPAID"
    try:
        response = khqr.check_payment(md5_hash)
        if response == "PAID":
            payment_status = "PAID"
    except Exception as e:
        logger.error(f"Bakong API Check Error: {e}")

    # 4. DEV OVERRIDE (The Fix)
    # If force=True is passed in the URL, we ignore the API and mark it PAID
    if force:
        logger.info(f"Force-approving transaction: {md5_hash}")
        payment_status = "PAID"

    # 5. Process the Success
    if payment_status == "PAID": 
        # Mark as completed in DB
        await transactions_collection.update_one(
            {"_id": trx["_id"]}, 
            {"$set": {"status": "COMPLETED", "paid_at": datetime.now().isoformat()}}
        )
        
        # Add Credits to User
        await add_credits(
            trx["user_id"], 
            trx["amount"], 
            f"Purchased Credits (Ref: {trx.get('payment_ref', 'N/A')})", 
            md5_hash
        )
        
        # Return new balance
        updated_user = await users_collection.find_one({"_id": trx["user_id"]})
        return {
            "status": "PAID", 
            "new_credits": trx["amount"], 
            "total_credits": updated_user.get("current_credits", 0)
        }
    
    return {"status": "UNPAID", "detail": "Payment not received yet"}

@app.get("/admin/transactions")
async def get_all_transactions(current_user: dict = Depends(get_current_user)):
    """
    Fetches the last 50 transactions for the Admin Panel.
    """
    # In a real app, add check: if current_user["role"] != "admin": raise ...
    
    cursor = transactions_collection.find().sort("created_at", -1).limit(50)
    transactions = []
    
    async for trx in cursor:
        # Get username for context
        user = await users_collection.find_one({"_id": trx["user_id"]})
        username = user["username"] if user else "Unknown"
        
        transactions.append({
            "id": str(trx["_id"]),
            "username": username,
            "amount": trx["amount"],
            "type": trx["type"],
            "status": trx["status"], # PENDING, COMPLETED
            "md5_hash": trx.get("md5_hash"),
            "payment_ref": trx.get("payment_ref"),
            "created_at": trx["created_at"]
        })
        
    return transactions

@app.delete("/admin/transactions")
async def clear_all_transactions(current_user: dict = Depends(get_current_user)):
    """
    Dev Utility: Wipes the entire transaction history.
    """
    await transactions_collection.delete_many({})
    return {"status": "success", "message": "Transaction history wiped."}

@app.post("/admin/add-credits")
async def admin_add_credits(username: str, amount: int):
    """
    Cheat code to instantly add credits to any user.
    """
    user = await users_collection.find_one({"username": username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Use your existing helper function so it logs the transaction
    await add_credits(
        user_id=user["_id"],
        amount=amount,
        reason="Admin Manual Top-up",
        ref="DEV_CHEAT"
    )
    
    return {"status": "success", "message": f"Added {amount} credits to {username}"}

