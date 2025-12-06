import os
import time
from typing import List, Optional
from datetime import datetime, timedelta, timezone  
import re
import tempfile
import asyncio
import json

# Web Framework
from fastapi import FastAPI, UploadFile, File, Query, BackgroundTasks, Response, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import httpx 
from apscheduler.schedulers.background import BackgroundScheduler

# Security
import bcrypt
from passlib.context import CryptContext
from jose import JWTError, jwt
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
import uuid
from bakong_khqr import KHQR

# Database
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

# Cloud Services
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
import google.generativeai as genai

load_dotenv()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

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
# users_collection = db.users  # New collection for Users
users_collection = db["users"]

transactions_collection = db["transactions"]
BAKONG_DEV_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYXRhIjp7ImlkIjoiZmU2MzNjNDdlOGZlNDQ0YSJ9LCJpYXQiOjE3NjQ5NTIyNDksImV4cCI6MTc3MjcyODI0OX0.tbhgtVlzNrTGhD0mKkN33BgopmENupueM7qa9DsDxOI"

BAKONG_ACCOUNT_ID = "say_vathanak@aclb"
MERCHANT_NAME = "SAY SAKSOVATHANAK"
MERCHANT_CITY = "Phnom Penh"

# Initialize the SDK once
khqr = KHQR(BAKONG_DEV_TOKEN) 
# ---------------------

# Auth Config
SECRET_KEY = os.getenv("SECRET_KEY", "YOUR_SUPER_SECRET_KEY_CHANGE_THIS_IN_PROD")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

# --- BACKGROUND SCHEDULER: AUTO-DELETE ---
def auto_delete_old_files():
    """
    Checks for files older than 24 hours and deletes them from Cloudinary
    to comply with privacy rules.
    """
    print(f"[Auto-Cleanup] Running cleanup job at {datetime.now()}...")
    
    # 1. Calculate the cutoff time (24 hours ago)
    cutoff_time = datetime.now() - timedelta(hours=24)
    cutoff_str = cutoff_time.isoformat()

    # 2. Find candidates that:
    #    - Have a 'cv_url' (file exists)
    #    - Were uploaded BEFORE the cutoff time
    #    - Are NOT already marked as 'Expired'
    #    (Note: In a full production app, you would also check the user's specific setting)
    query = {
        "upload_date": {"$lt": cutoff_str},
        "cv_url": {"$ne": None},
        "file_status": {"$ne": "Expired"} # We will add this field
    }

    # We need to run this inside an async loop since Motor is async
    # But APScheduler is sync. So we use a little helper or just run sync logic if possible.
    # Since Motor is strictly async, we'll swap to a simple synchronous loop for the scheduler 
    # OR simpler: Trigger it manually for this MVP. 
    
    # FOR SIMPLICITY in this code snippet, we will print what WOULD happen.
    # To make this robust with AsyncIOMotorClient, we usually attach it to the FastAPI startup event.
    pass 

# Since combining Async Mongo with Sync Scheduler is tricky in one file, 
# here is the ASYNC version you can call from a startup event.
async def run_async_cleanup():
    print("[Auto-Cleanup] Scanning for old files...")
    cutoff_time = datetime.now() - timedelta(hours=24)
    
    # 1. THE QUERY
    # We ask for candidates uploaded > 24h ago
    # AND where cv_url is NOT null
    cursor = collection.find({
        "upload_date": {"$lt": cutoff_time.isoformat()},
        "cv_url": {"$ne": None}, 
        "file_status": {"$ne": "Expired"}
    })

    async for candidate in cursor:
        url = candidate.get("cv_url")
        
        # 2. EXTRA SAFETY CHECK (Python Level)
        # If url is missing, empty, or just text like "Manual Entry", SKIP IT.
        if not url or "http" not in url:
            continue 

        try:
            # 3. Extract ID and Delete
            public_id = url.split('/')[-1].split('.')[0]
            print(f" -> Deleting PDF for: {candidate.get('Name')}")
            
            cloudinary.uploader.destroy(public_id)
            
            # 4. Update Status
            await collection.update_one(
                {"_id": candidate["_id"]},
                {"$set": {
                    "cv_url": None, # Clear the URL so we don't try again
                    "file_status": "Expired"
                }}
            )
        except Exception as e:
            print(f"Error cleaning {candidate.get('_id')}: {e}")

# App Config
app = FastAPI()

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://cvtracker-kh.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AUTH MODELS ---
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class BulkDeleteRequest(BaseModel):
    candidate_ids: List[str] = []
    # Removed passcode since we now use JWT Auth
    
class GoogleAuthRequest(BaseModel):
    token: str
    
class PaymentRequest(BaseModel):
    package_id: str
    email: str

@app.post("/auth/google")
async def google_login(request: GoogleAuthRequest):
    try:
        # 1. Verify the token with Google
        id_info = id_token.verify_oauth2_token(
            request.token, 
            google_requests.Request(), 
            GOOGLE_CLIENT_ID
        )

        # 2. Extract user info
        email = id_info.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Invalid Google Token: No email found")

        # 3. Check if user exists in DB
        user = await users_collection.find_one({"username": email})

        if not user:
            # 4. If not, Register them automatically
            new_user = {
                "username": email,
                "hashed_password": "GOOGLE_AUTH_USER",
                "provider": "google",
                "created_at": datetime.now().isoformat(),
                "current_credits": 0 
            }
            
            result = await users_collection.insert_one(new_user)
            
            # --- GIVE FREE 10 CREDITS ---
            await add_credits(
                user_id=result.inserted_id, 
                amount=10, 
                reason="Welcome Gift: Free 10 Credits",
                ref="SIGNUP_BONUS"
            )
        
        # 5. Generate Access Token (Log them in)
        # CRITICAL FIX: Ensure this is indented to match the 'try', NOT the 'if'
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": email}, expires_delta=access_token_expires
        )
        
        return {"access_token": access_token, "token_type": "bearer", "username": email}

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Google Token")
    except Exception as e:
        print(f"Google Auth Error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")
    
# --- LIFECYCLE EVENTS ---
@app.on_event("startup")
async def start_scheduler():
    # Initialize the scheduler
    scheduler = BackgroundScheduler()
    
    # Add the job (Run every 60 minutes)
    # We wrap the async function in a sync wrapper or just use a loop
    # For simplicity, we will just run the cleanup ONCE on startup 
    # and then you can add a specialized endpoint to trigger it manually or via cron.
    
    print("--> System Startup: Checking for expired files...")
    await run_async_cleanup()
    
    # Start the scheduler for future ticks (requires slightly more setup for async)
    # scheduler.add_job(some_sync_wrapper, 'interval', minutes=60)
    # scheduler.start()

# --- SECURITY HELPERS ---
def verify_password(plain_password, hashed_password):
    # Fix: Ensure the plain_password is a byte string and is truncated if necessary,
    # to prevent the bcrypt ValueError.
    if isinstance(plain_password, str):
        # The password needs to be a byte string, encoded using utf-8 is standard
        # and then truncated to 72 bytes (characters *might* be less than 72).
        plain_password_bytes = plain_password.encode('utf-8')[:72]
    else:
        # If it's already bytes, truncate it.
        plain_password_bytes = plain_password[:72]
        
    # The 'pwd_context.verify' method handles the actual verification
    return pwd_context.verify(plain_password_bytes, hashed_password)

def get_password_hash(password):
    # It is good practice to do the same truncation on hashing, 
    # to ensure consistency between stored hashes and verification.
    if isinstance(password, str):
        password_bytes = password.encode('utf-8')[:72]
    else:
        password_bytes = password[:72]
        
    return pwd_context.hash(password_bytes)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = await users_collection.find_one({"username": username})
    if user is None:
        raise credentials_exception
    return user

async def process_cv_background(file_content: bytes, filename: str, cv_url: str, candidate_id: str, mime_type: str):
    """
    Runs in background: Uploads to Gemini -> Extracts Data -> Updates MongoDB
    Refactored for MAXIMUM SPEED (No Semaphore/Throttling).
    """
    temp_path = None
    gemini_file = None
    
    try:
        print(f"[{filename}] 🚀 Started processing (Parallel Mode)...")

        # 1. Save bytes to a temp file (Gemini requirement)
        suffix = ".pdf" if mime_type == "application/pdf" else ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            temp_path = tmp.name

        # 2. Upload to Gemini Files API
        gemini_file = genai.upload_file(path=temp_path, mime_type=mime_type)

        # 3. Wait for processing (Poll until Active)
        # We keep the sleep small (1s) to be responsive
        while gemini_file.state.name == "PROCESSING":
            await asyncio.sleep(1)
            gemini_file = genai.get_file(gemini_file.name)

        if gemini_file.state.name == "FAILED":
            raise Exception("Gemini failed to process the file media.")

        # 4. Generate Content (The Extraction)
        # Using 'gemini-1.5-flash' (or 2.0) is recommended for speed
        model = genai.GenerativeModel('gemini-2.0-flash', generation_config={"response_mime_type": "application/json"})
        
        prompt = """
        You are an expert HR Data Extractor for candidates in Cambodia.
        Analyze the uploaded CV and extract details into a JSON object.

        ### 1. STANDARDIZATION RULES (Strict Enums):
        - **EducationLevel:** MUST be one of: ['High School', 'Associate Degree', 'Bachelor Degree', 'Master Degree', 'PhD', 'Other']. 
          (Map 'B.Sc', 'Year 3 Student', 'University' -> 'Bachelor Degree').
        - **Gender:** MUST be one of: ['Male', 'Female', 'N/A'].
        
        ### 2. LOCATION RULES (Cambodia Context):
        - Format: "Sangkat [Name], Khan [Name]" or "City, Province".
        - If only "Phnom Penh" is found, return "Phnom Penh".

        ### 3. CONFIDENCE SCORE (0-100):
        - Rate your confidence in the extraction accuracy.
        - **100:** Perfect PDF, clear text, all fields found.
        - **80:** Good, but maybe missing Address or Birthday.
        - **50:** Blurry image, handwriting, or very sparse data.
        - **0:** Unreadable.

        ### 4. DATA FIELDS:
        - **Name:** Full Name (Title Case). Remove "Mr./Ms.".
        - **Tel:** "0xx xxx xxx" format. Remove (+855).
        - **Experience:** Summarize last job title & company (Max 15 words).
        - **Position:** The role they are applying for (or current role).
        - **School:** Name of the University/School only.

        Return JSON with keys:
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
        
        # 5. Clean & Parse JSON
        try:
            # Strip code blocks if Gemini adds them
            json_text = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(json_text)
            
            # Handle edge case where AI returns a list [ {data} ]
            if isinstance(data, list):
                data = data[0] if len(data) > 0 else {}
        except Exception as parse_error:
            print(f"[{filename}] JSON Parse Error: {parse_error}")
            data = {"Name": "Parse Error", "Confidence": 0, "Experience": "AI output invalid JSON"}

        # 6. Update MongoDB with Real Data
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
            "status": "Ready",  # MARK AS DONE
            "last_modified": datetime.now().isoformat()
        }

        # Assuming 'collection' is your MongoDB collection object defined globally or imported
        await collection.update_one(
            {"_id": ObjectId(candidate_id)}, 
            {"$set": update_payload}
        )
        print(f"[{filename}] ✅ Success! Confidence: {data.get('Confidence')}")

    except Exception as e:
        print(f"[{filename}] ❌ Error: {e}")
        await collection.update_one(
            {"_id": ObjectId(candidate_id)}, 
            {"$set": {"status": "Error", "error_msg": str(e)}}
        )

    finally:
        # --- CLEANUP (CRITICAL FOR PRIVACY & STORAGE) ---
        # 1. Delete local temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        
        # 2. Delete file from Gemini Cloud (Save storage space)
        if gemini_file:
            try:
                genai.delete_file(gemini_file.name)
                print(f"[{filename}] Cleaned up Gemini file.")
            except Exception as e:
                print(f"[{filename}] Cleanup Warning: {e}")

# --- AUTH ENDPOINTS ---

@app.post("/register", status_code=201)
async def register(user: UserCreate):
    # Check if user exists
    existing_user = await users_collection.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Hash password and save
    hashed_pw = get_password_hash(user.password)
    new_user = {"username": user.username, "hashed_password": hashed_pw}
    await users_collection.insert_one(new_user)
    return {"message": "User created successfully"}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # Find user
    user = await users_collection.find_one({"username": form_data.username})
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate Token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    # Return both username AND current_credits
    return {
        "username": current_user["username"],
        "current_credits": current_user.get("current_credits", 0) 
    }

async def add_credits(user_id, amount: int, reason: str, ref: str = None):
    """
    Updates user balance AND records the transaction history.
    """
    # 1. Create a transaction record
    trx = {
        "user_id": user_id, # Stores the ObjectId of the user
        "amount": amount,
        "type": "PURCHASE" if amount > 0 else "SPEND",
        "description": reason,
        "payment_ref": ref,
        "created_at": datetime.now().isoformat(),
        "status": "COMPLETED"
    }
    await transactions_collection.insert_one(trx)

    # 2. Update the User's "current_credits" balance
    await users_collection.update_one(
        {"_id": user_id},
        {"$inc": {"current_credits": amount}}
    )

# --- CORE API ENDPOINTS ---

@app.post("/admin/cleanup")
async def trigger_cleanup(current_user: dict = Depends(get_current_user)):
    # Security: Only allow if user is Admin (or just any logged in user for now)
    await run_async_cleanup()
    return {"status": "Cleanup job finished."}

@app.post("/upload-cv")
async def upload_cv(
    background_tasks: BackgroundTasks, 
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)  # PROTECTED
):
    results = []
    
    # --- 1. CREDIT CHECK & DEDUCTION ---
    cost = len(files) # 1 Credit per file
    user_credits = current_user.get("current_credits", 0)
    
    # A. Check if user has enough credits
    if user_credits < cost:
        raise HTTPException(
            status_code=402, # 402 Payment Required
            detail=f"Insufficient credits. You have {user_credits}, but need {cost}."
        )

    # B. Deduct credits immediately (Atomic update)
    await users_collection.update_one(
        {"_id": current_user["_id"]},
        {"$inc": {"current_credits": -cost}}
    )
    
    # C. Record the 'Spend' transaction
    await transactions_collection.insert_one({
        "user_id": current_user["_id"],
        "amount": -cost,
        "type": "SPEND",
        "description": f"Uploaded {cost} CV(s)",
        "created_at": datetime.now().isoformat(),
        "status": "COMPLETED"
    })

    # --- 2. PROCESS FILES ---
    for file in files:
        try:
            # Read file into memory
            content = await file.read()
            mime_type = file.content_type or "application/pdf"
            
            # Upload to Cloudinary (Sync)
            clean_name = re.sub(r'[^a-zA-Z0-9]', '_', file.filename.split('.')[0])
            upload_result = cloudinary.uploader.upload(content, resource_type="auto", public_id=clean_name)
            cv_url = upload_result.get("secure_url")
            
            # Create "Placeholder" Candidate
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
                "status": "Processing",
                "uploaded_by": current_user["username"] 
            }
            
            insert_result = await collection.insert_one(placeholder_data)
            candidate_id = str(insert_result.inserted_id)
            
            # Trigger Background Task
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
            
    # Return success along with remaining credits so frontend knows
    return {
        "status": f"Queued {len(results)} files", 
        "details": results,
        "remaining_credits": user_credits - cost
    }

@app.get("/candidates")
async def get_candidates(
    page: int = Query(1, ge=1), 
    limit: int = Query(20, le=100), 
    search: str = Query(None),
    current_user: dict = Depends(get_current_user) # <--- NEW: Require Login
):
    # 1. Base Filter: Only show candidates uploaded by THIS user
    query_filter = {"uploaded_by": current_user["username"]}

    # 2. Add Search Logic (if user is typing)
    if search:
        search_regex = {"$regex": search, "$options": "i"}
        # Use $and to ensure we match the User AND the Search term
        query_filter = {
            "$and": [
                {"uploaded_by": current_user["username"]},
                {
                    "$or": [
                        {"Name": search_regex}, {"Tel": search_regex},
                        {"School": search_regex}, {"Location": search_regex}
                    ]
                }
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
    # Keeping this public for previewing, but you can protect it if needed.
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
async def delete_candidate(
    candidate_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        obj_id = ObjectId(candidate_id)
        
        # CHANGED: Check if candidate exists AND belongs to the user
        candidate = await collection.find_one({
            "_id": obj_id, 
            "uploaded_by": current_user["username"]
        })
        
        if not candidate:
            return {"status": "Error: Candidate not found or access denied"}
            
        if candidate.get("locked", False): 
            return {"status": "Error: Candidate is locked"}
            
        await collection.delete_one({"_id": obj_id})
        return {"status": "Deleted successfully"}
    except Exception as e:
        return {"status": f"Error: {e}"}

@app.put("/candidates/{candidate_id}")
async def update_candidate(
    candidate_id: str, 
    updated_data: dict,
    current_user: dict = Depends(get_current_user)
):
    try:
        if "_id" in updated_data: del updated_data["_id"]
        updated_data["last_modified"] = datetime.now().isoformat()
        
        # CHANGED: Add 'uploaded_by' to the query
        result = await collection.update_one(
            {"_id": ObjectId(candidate_id), "uploaded_by": current_user["username"]}, 
            {"$set": updated_data}
        )
        
        if result.matched_count == 0:
            return {"status": "Error: Candidate not found or access denied"}
            
        return {"status": "Updated successfully"}
    except Exception as e:
        return {"status": f"Error: {e}"}

@app.put("/candidates/{candidate_id}/lock")
async def toggle_lock(
    candidate_id: str, 
    request: dict,
    current_user: dict = Depends(get_current_user)
):
    try:
        # CHANGED: Add 'uploaded_by' to the query
        result = await collection.update_one(
            {"_id": ObjectId(candidate_id), "uploaded_by": current_user["username"]}, 
            {"$set": {"locked": request.get("locked", False)}}
        )
        
        if result.matched_count == 0:
            return {"status": "Error: Access denied"}

        return {"status": "success"}
    except Exception as e:
        return {"status": f"Error: {e}"}

# --- BULK DELETE & RETRY ENDPOINTS ---

@app.post("/candidates/bulk-delete")
async def bulk_delete_candidates(
    request: BulkDeleteRequest,
    current_user: dict = Depends(get_current_user)
):
    try:
        user_filter = {"uploaded_by": current_user["username"], "locked": {"$ne": True}}

        # SCENARIO 1: DELETE ALL (Only user's own data)
        if not request.candidate_ids:
            result = await collection.delete_many(user_filter)
            return {"status": "success", "message": f"Wiped {result.deleted_count} records"}

        # SCENARIO 2: DELETE SELECTED
        elif request.candidate_ids:
            ids_to_delete = []
            for cid in request.candidate_ids:
                try: ids_to_delete.append(ObjectId(cid))
                except: continue
            
            # Add ID filter to the User filter
            user_filter["_id"] = {"$in": ids_to_delete}
            
            result = await collection.delete_many(user_filter)
            return {"status": "success", "message": f"Deleted {result.deleted_count} candidates"}

        return {"status": "error", "message": "Invalid request"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.post("/candidates/{candidate_id}/retry")
async def retry_parsing(
    candidate_id: str, 
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    try:
        # CHANGED: Check ownership
        candidate = await collection.find_one({
            "_id": ObjectId(candidate_id),
            "uploaded_by": current_user["username"]
        })
        
        if not candidate:
            return {"status": "error", "message": "Candidate not found or access denied"}

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
    
@app.post("/api/create-payment")
async def create_khqr_payment(request: PaymentRequest):
    package_id = request.package_id
    email = request.email

    packages = {
        "small": {"price": 1.00, "credits": 20}, 
        "pro":   {"price": 5.00, "credits": 150} 
    }
    
    if package_id not in packages:
        raise HTTPException(status_code=400, detail="Invalid package")
    
    pkg = packages[package_id]
    
    # 1. Generate unique bill number
    bill_number = str(uuid.uuid4().int)[:10] 

    # 2. Create the QR String
    qr_response = khqr.create_qr(
        bank_account=BAKONG_ACCOUNT_ID,
        merchant_name=MERCHANT_NAME,
        merchant_city=MERCHANT_CITY,
        amount=pkg["price"],
        currency="USD", 
        phone_number='85592886006',
        store_label="CV Credits",
        bill_number=bill_number,
        terminal_label="POS-01"
    )
    
    # ⚠️ NEW STEP: Generate the MD5 Hash for tracking
    # The library returns a dictionary for create_qr usually, but your sample implies string.
    # If qr_response is a dictionary (common in some versions), use qr_response['qr']
    # If qr_response is just a string (as per your sample), use it directly.
    qr_string = qr_response 
    md5_hash = khqr.generate_md5(qr_string)

    # 3. Save PENDING transaction with MD5
    user = await users_collection.find_one({"username": email})
    if user:
        await transactions_collection.insert_one({
            "user_id": user["_id"],
            "amount": pkg["credits"],
            "type": "PURCHASE_INTENT",
            "status": "PENDING",
            "payment_ref": bill_number,
            "md5_hash": md5_hash,  # <--- WE SAVE THIS NOW
            "created_at": datetime.now().isoformat()
        })

    return {
        "qr_code": qr_string, 
        "md5": md5_hash,
        "amount": pkg["price"]
    }
    
@app.post("/api/check-payment-status")
async def check_payment_status(md5_hash: str):
    """
    Checks if a specific transaction was paid using the Bakong API.
    """
    # 1. Find the transaction in OUR database
    trx = await transactions_collection.find_one({"md5_hash": md5_hash})
    
    if not trx:
        raise HTTPException(status_code=404, detail="Transaction not found")
        
    if trx["status"] == "COMPLETED":
        return {"status": "PAID", "message": "Already processed"}

    # 2. Call Bakong API to verify
    # Note: This requires BAKONG_DEV_TOKEN to be valid!
    try:
        # Based on your sample, this returns "UNPAID" or "PAID"
        payment_status = khqr.check_payment(md5_hash)
    except Exception as e:
        # If token is invalid or API is down
        print(f"Bakong Error: {e}")
        raise HTTPException(status_code=500, detail="Cannot verify payment with Bank")

    # 3. If Paid, give credits
    if payment_status == "PAID": 
        # Update transaction status
        await transactions_collection.update_one(
            {"_id": trx["_id"]},
            {"$set": {"status": "COMPLETED", "paid_at": datetime.now().isoformat()}}
        )
        
        # Add Credits to User
        await add_credits(
            user_id=trx["user_id"],
            amount=trx["amount"],
            reason=f"Purchased Credits (Ref: {trx['payment_ref']})",
            ref=md5_hash
        )
        return {"status": "PAID", "new_credits": trx["amount"]}
    
    else:
        return {"status": "UNPAID", "detail": "Payment not received yet"}