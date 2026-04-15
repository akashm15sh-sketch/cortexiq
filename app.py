"""CortexIQ Neural Signal Prototype — FastAPI Backend."""
import os
import json
import shutil
import uuid
import random
import hashlib
import numpy as np
from datetime import datetime, timezone
from typing import Optional, Dict, List

def _sanitize(obj):
    """Recursively convert numpy types to Python native types for JSON serialization.
    Also replaces nan/inf with None to ensure JSON compliance."""
    import math
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    elif isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import iterate_in_threadpool
from pydantic import BaseModel

from cortexiq.database import init_db, SessionLocal, User, Study, AnalysisJob, ChatMessage
from cortexiq.config import UPLOAD_DIR, RESULTS_DIR, POSTHOG_API_KEY, POSTHOG_HOST
from cortexiq.auth.jwt_handler import create_token, verify_token
from cortexiq.ai.interpreter import CortexIQInterpreter
from cortexiq.eeg.loader import EEGLoader
from cortexiq.eeg.pipeline import EEGPipeline
from cortexiq.eeg.reporter import EEGReporter

# ── PostHog Server-Side Analytics ──
try:
    from posthog import Posthog
    if POSTHOG_API_KEY:
        posthog_client = Posthog(POSTHOG_API_KEY, host=POSTHOG_HOST)
        logger.info("PostHog analytics initialized.")
    else:
        posthog_client = None
        logger.info("PostHog API key not set — analytics disabled.")
except ImportError:
    posthog_client = None
    logger.warning("posthog-python not installed — analytics disabled. Install with: pip install posthog")


def _ph(event: str, user_id, properties: dict = None):
    """Fire-and-forget PostHog event capture. Silently skips if PostHog is not configured."""
    if not posthog_client:
        return
    try:
        posthog_client.capture(
            distinct_id=str(user_id),
            event=event,
            properties=properties or {}
        )
    except Exception as e:
        logger.debug(f"PostHog tracking error: {e}")


def _ph_identify(user_id, properties: dict = None):
    """Identify a user in PostHog with their properties."""
    if not posthog_client:
        return
    try:
        posthog_client.identify(
            distinct_id=str(user_id),
            properties=properties or {}
        )
    except Exception as e:
        logger.debug(f"PostHog identify error: {e}")

# ── Init ──
init_db()
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(title="CortexIQ Neural Engine", version="0.2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ──
ai_interpreter = CortexIQInterpreter()
eeg_loader = EEGLoader()
eeg_reporter = EEGReporter()

# Mount results directory as static files for serving figures
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Per-User Session Store ──
# subjects: [{"id": 0, "name": "...", "raw": ..., "info": ...}, ...]
MAX_SESSIONS = 50
user_sessions: Dict[int, dict] = {}
otp_store = {}  # email -> {"code": "123456", "expires": float}

# ── Helpers ──
def hash_password(password: str) -> str: return hashlib.sha256(password.encode()).hexdigest()
def check_password(password: str, hashed: str) -> bool: return hash_password(password) == hashed

def get_user_session(user_id: int) -> dict:
    if user_id not in user_sessions:
        if len(user_sessions) >= MAX_SESSIONS:
            oldest_key = next(iter(user_sessions))
            del user_sessions[oldest_key]
        user_sessions[user_id] = {
            "subjects": [],
            "study_id": None,
            "study_ctx": {},
            "chat_history": [],
            "pending_steps": [],
            "pipeline": EEGPipeline(),
        }
    return user_sessions[user_id]

def get_user_pipeline(user_id: int) -> EEGPipeline:
    session = get_user_session(user_id)
    if "pipeline" not in session:
        session["pipeline"] = EEGPipeline()
    return session["pipeline"]

# ── Pydantic models ──
class SendOTPRequest(BaseModel): email: str
class VerifyOTPRequest(BaseModel): email: str; code: str
class RegisterRequest(BaseModel): email: str; otp_code: str; username: str; password: str
class LoginRequest(BaseModel): username: str; password: str; remember_me: bool = False
class UpdateProfileRequest(BaseModel): google_scholar: str = ""; institution: str = ""; bio: str = ""
class StudyCreateRequest(BaseModel):
    name: str
    modality: str = "EEG"
    subject_count: int = 1
    conditions: str = ""
    sfreq: float = 256.0
    reference: str = "average"
    notes: str = ""
    montage: List[str] = []
    events: Optional[dict] = None
class ChatRequest(BaseModel):
    message: str
    model: str = "claude"


# ── Auth dependency ──
def get_current_user(request: Request) -> dict:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "): raise HTTPException(status_code=401, detail="Not authenticated")
    payload = verify_token(auth.split(" ")[1])
    if not payload: raise HTTPException(status_code=401, detail="Invalid token")
    return payload


# ═══════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════

from cortexiq.utils.email import send_otp_email

@app.post("/api/auth/send-otp")
async def send_otp(req: SendOTPRequest):
    email = req.email.strip().lower()
    db = SessionLocal()
    if db.query(User).filter(User.email == email).first():
        db.close(); raise HTTPException(status_code=400, detail="Email exists. Please login.")
    db.close()
    code = f"{random.randint(100000, 999999)}"
    otp_store[email] = {"code": code, "expires": datetime.now(timezone.utc).timestamp() + 300}
    
    # Send real email
    success = send_otp_email(email, code)
    
    if success:
        return {"message": f"Verification code sent to {email}", "status": "sent"}
    else:
        return {"message": "Email delivery unavailable. Use the code shown below.", "otp_code": code, "status": "fallback"}

@app.post("/api/auth/verify-otp")
async def verify_otp(req: VerifyOTPRequest):
    email, stored = req.email.strip().lower(), otp_store.get(req.email.strip().lower())
    if not stored or datetime.now(timezone.utc).timestamp() > stored["expires"]: raise HTTPException(status_code=400, detail="OTP expired.")
    if stored["code"] != req.code.strip(): raise HTTPException(status_code=400, detail="Invalid OTP.")
    return {"verified": True}

@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    email, username = req.email.strip().lower(), req.username.strip()
    if len(username) < 3 or len(req.password) < 6: raise HTTPException(status_code=400, detail="Short username/password.")
    stored = otp_store.get(email)
    if not stored or datetime.now(timezone.utc).timestamp() > stored["expires"]: raise HTTPException(status_code=400, detail="OTP expired.")
    if stored["code"] != req.otp_code.strip(): raise HTTPException(status_code=400, detail="Invalid OTP.")
    db = SessionLocal()
    if db.query(User).filter((User.email == email) | (User.username == username)).first():
        db.close(); raise HTTPException(status_code=400, detail="User exists.")
    user = User(email=email, username=username, password_hash=hash_password(req.password), tier="Researcher", login_count=1)
    db.add(user); db.commit(); u_id = user.id; db.close()
    if email in otp_store: del otp_store[email]
    _ph_identify(u_id, {"email": email, "username": username, "tier": "Researcher"})
    _ph("user_registered", u_id, {"email": email, "username": username})
    return {"token": create_token(u_id, username, email, "Researcher"), "user_id": u_id, "username": username, "email": email}

@app.post("/api/auth/login")
async def login(req: LoginRequest):
    db = SessionLocal()
    u = db.query(User).filter((User.username == req.username) | (User.email == req.username)).first()
    if not u or not check_password(req.password, u.password_hash):
        _ph("login_failed", req.username, {"attempted_username": req.username})
        db.close(); raise HTTPException(status_code=401, detail="Invalid credentials.")
    u.login_count += 1; u.last_login = datetime.now(timezone.utc); db.commit()
    res = {"token": create_token(u.id, u.username, u.email, u.tier, req.remember_me), "user_id": u.id, "username": u.username, "email": u.email, "tier": u.tier, "login_count": u.login_count}
    _ph_identify(u.id, {"email": u.email, "username": u.username, "tier": u.tier, "login_count": u.login_count})
    _ph("user_logged_in", u.id, {"username": u.username, "login_count": u.login_count, "remember_me": req.remember_me})
    db.close(); return res

@app.get("/api/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    db = SessionLocal(); u = db.query(User).filter(User.id == int(user["sub"])).first()
    if not u: db.close(); raise HTTPException(status_code=404, detail="User not found")
    res = {"user_id": u.id, "username": u.username, "email": u.email, "tier": u.tier, "google_scholar": u.google_scholar or "", "institution": u.institution or "", "login_count": u.login_count}
    db.close(); return res

@app.put("/api/user/profile")
async def update_profile(req: UpdateProfileRequest, user: dict = Depends(get_current_user)):
    db = SessionLocal(); u = db.query(User).filter(User.id == int(user["sub"])).first()
    if u: u.google_scholar, u.institution, u.bio = req.google_scholar, req.institution, req.bio; db.commit()
    _ph("profile_updated", user["sub"], {"has_scholar": bool(req.google_scholar), "has_institution": bool(req.institution)})
    db.close(); return {"message": "Profile updated"}

@app.get("/api/user/history")
async def get_history(user: dict = Depends(get_current_user)):
    db = SessionLocal(); uid = int(user["sub"])
    studies = db.query(Study).filter(Study.user_id == uid).order_by(Study.created_at.desc()).limit(10).all()
    res = {"studies": [{"id": s.id, "name": s.name, "modality": s.modality, "created_at": s.created_at.isoformat()} for s in studies]}
    db.close(); return res


# ═══════════════════════════════════════════════════
# ANALYSIS ENDPOINTS
# ═══════════════════════════════════════════════════

@app.post("/api/study/upload")
async def upload_files(files: List[UploadFile] = File(...), sfreq: float = Form(256.0), user: dict = Depends(get_current_user)):
    uid = int(user["sub"]); session = get_user_session(uid)
    new_subjects = []
    
    for file in files:
        f_id = f"{uuid.uuid4().hex[:6]}"
        save_path = os.path.join(UPLOAD_DIR, f"u{uid}_{f_id}{os.path.splitext(file.filename or 'unknown')[1].lower()}")
        with open(save_path, "wb") as f: f.write(await file.read())
        
        raw, info = eeg_loader.load(save_path, sfreq=sfreq)
        if raw is None:
            if os.path.exists(save_path): os.remove(save_path)
            continue # Skip failed files
            
        # Preview data (all channels)
        n_show = info.n_channels
        n_samples = int(min(5.0, info.duration_sec) * info.sfreq)
        data = raw.get_data(stop=n_samples)
        
        # Demean each channel to remove DC offsets
        for i in range(data.shape[0]):
            ch_mean = np.mean(data[i])
            data[i] -= ch_mean
            
        times = np.linspace(0, n_samples/info.sfreq, n_samples).tolist()
        
        # Calculate robust vertical spacing based on standard deviation
        # Use median of per-channel std to avoid outliers (e.g. noisy channels)
        stds = np.std(data, axis=1)
        std_val = float(np.median(stds)) if len(stds) > 0 and np.median(stds) > 0 else 1e-6
        if std_val < 1e-8: std_val = 1e-6 # fallback for extremely flat data
        
        ch_pre = []
        for i in range(n_show):
            # i=0 at top for preview
            offset = float((n_show - 1 - i) * std_val * 6)
            ch_pre.append({"name": info.channel_names[i], "data": [float(v) + offset for v in data[i].tolist()]})
            
        subj = {"id": len(session["subjects"]), "name": file.filename or f"upload_{f_id}", "path": save_path, "raw": raw, "info": info, "preview": {"times": times, "channels": ch_pre}}
        session["subjects"].append(subj)
        new_subjects.append({"id": subj["id"], "name": subj["name"], "path": save_path, "info": {"format": info.format_name, "n_channels": info.n_channels, "sfreq": info.sfreq, "duration_sec": round(info.duration_sec, 1)}, "preview": subj["preview"]})

    if not new_subjects: raise HTTPException(status_code=400, detail="Supported files not found.")
    _ph("file_uploaded", user["sub"], {"n_files": len(new_subjects), "sfreq": sfreq, "filenames": [s["name"] for s in new_subjects], "n_channels": new_subjects[0]["info"]["n_channels"] if new_subjects else 0})
    return {"subjects": new_subjects}

@app.get("/api/study/session")
async def get_study_session(user: dict = Depends(get_current_user)):
    uid = int(user["sub"]); session = get_user_session(uid)
    # Return serializable summary of subjects (include channel_names for frontend mapping list)
    subjs = []
    for s in session["subjects"]:
        subjs.append({
            "id": s["id"], "name": s["name"], 
            "info": {
                "format": s["info"].format_name,
                "n_channels": s["info"].n_channels,
                "channel_names": list(s["info"].channel_names),
                "sfreq": s["info"].sfreq,
                "duration_sec": round(s["info"].duration_sec, 1)
            }, 
            "preview": s["preview"]
        })
    return {"subjects": subjs, "study_ctx": session.get("study_ctx", {})}


@app.get("/api/study/data/{subject_id}")
async def get_subject_data(subject_id: int, tmin: float = 0.0, tmax: float = 30.0, user: dict = Depends(get_current_user)):
    """Fetch EEG time-series data for a subject within [tmin, tmax] seconds."""
    uid = int(user["sub"]); session = get_user_session(uid)
    subjects = session.get("subjects", [])
    subj = next((s for s in subjects if s["id"] == subject_id), None)
    if not subj:
        raise HTTPException(status_code=404, detail="Subject not found")
    raw = subj.get("raw")
    if raw is None:
        raise HTTPException(status_code=400, detail="Raw data not available for this subject")

    info = subj["info"]
    sfreq = info.sfreq
    total_dur = info.duration_sec
    tmin = max(0.0, tmin)
    tmax = min(total_dur, tmax)
    if tmax <= tmin:
        tmax = min(tmin + 1.0, total_dur)

    # Cap the chunk to 60s to avoid huge payloads
    if tmax - tmin > 60.0:
        tmax = tmin + 60.0

    start_sample = int(tmin * sfreq)
    stop_sample = int(tmax * sfreq)
    data = raw.get_data(start=start_sample, stop=stop_sample)
    
    # Demean each channel to remove DC offsets for visualization
    for i in range(data.shape[0]):
        ch_mean = np.mean(data[i])
        data[i] -= ch_mean
        
    times = np.linspace(tmin, tmax, data.shape[1]).tolist()

    # Calculate robust vertical spacing
    # Use median of per-channel std to avoid outliers (e.g. noisy channels)
    stds = np.std(data, axis=1)
    std_val = float(np.median(stds)) if len(stds) > 0 and np.median(stds) > 0 else 1e-6
    if std_val < 1e-8: std_val = 1e-6 # fallback

    montage = session.get("study_ctx", {}).get("montage", [])
    channels = []
    n_ch = info.n_channels
    for i in range(n_ch):
        # Ch0 (first column) gets the highest offset = top of plot
        offset = float((n_ch - 1 - i) * std_val * 6)
        electrode_name = montage[i] if i < len(montage) and montage[i] else info.channel_names[i]
        channels.append({
            "name": electrode_name, 
            "offset": offset, 
            "raw_name": info.channel_names[i], 
            "data": [float(v) + offset for v in data[i].tolist()]
        })

    return {"times": times, "channels": channels, "tmin": tmin, "tmax": tmax, "total_duration": total_dur}


@app.post("/api/study/create")
async def create_study(req: StudyCreateRequest, user: dict = Depends(get_current_user)):
    uid = int(user["sub"]); session = get_user_session(uid); subjects = session.get("subjects", [])
    if not subjects: raise HTTPException(status_code=400, detail="No files uploaded.")
    
    # Prepare files metadata for persistence
    files_meta = []
    for s in subjects:
        files_meta.append({
            "name": s["name"], 
            "path": s["path"],
            "info": {"format": s["info"].format_name, "n_channels": s["info"].n_channels, "sfreq": s["info"].sfreq, "duration_sec": s["info"].duration_sec}
        })

    db = SessionLocal()
    study = Study(
        user_id=uid, 
        name=req.name, 
        modality=req.modality, 
        subject_count=req.subject_count,
        n_channels=subjects[0]["info"].n_channels, 
        conditions=req.conditions, 
        sfreq=req.sfreq, 
        reference=req.reference,
        notes=req.notes,
        files_json=json.dumps(files_meta)
    )
    db.add(study); db.commit(); s_id = study.id; db.close()
    
    # Build per-subject summary
    subjects_summary = []
    total_duration = 0.0
    for s in subjects:
        info = s["info"]
        total_duration += info.duration_sec
        subjects_summary.append({
            "name": s["name"],
            "n_channels": info.n_channels,
            "channel_names": info.channel_names,
            "sfreq": info.sfreq,
            "duration_sec": round(info.duration_sec, 2),
            "format": info.format_name,
        })

    # Compute effective channel names: prefer user's montage assignments over raw file names
    raw_ch_names = subjects[0]["info"].channel_names
    montage = req.montage or []
    effective_channel_names = [
        (montage[i] if i < len(montage) and montage[i] else raw_ch_names[i])
        for i in range(len(raw_ch_names))
    ]
    # Build channel mapping list for the AI (raw→electrode) when montage differs
    channel_mapping = []
    for i, (raw, eff) in enumerate(zip(raw_ch_names, effective_channel_names)):
        if raw != eff:
            channel_mapping.append({"index": i + 1, "raw": raw, "label": eff})

    # Strip binary/non-serialisable content from events (ArrayBuffer from frontend)
    events_ctx = None
    if req.events:
        content = req.events.get("content")
        if isinstance(content, str):
            events_ctx = {
                "name": req.events.get("name", ""),
                "size": req.events.get("size", 0),
                "content": content,
            }
        else:
            events_ctx = {
                "name": req.events.get("name", ""),
                "size": req.events.get("size", 0),
            }

    session["study_id"] = s_id
    session["study_ctx"] = {
        "study_id": s_id,
        "name": req.name,
        "modality": req.modality,
        "n_subjects": len(subjects),
        "sfreq": req.sfreq,
        "n_channels": subjects[0]["info"].n_channels,
        "raw_channel_names": raw_ch_names,
        "channel_names": effective_channel_names,
        "channel_mapping": channel_mapping,
        "duration_sec": round(subjects[0]["info"].duration_sec, 2),
        "file_format": subjects[0]["info"].format_name,
        "conditions": req.conditions,
        "reference": req.reference,
        "notes": req.notes,
        "montage": req.montage,
        "events": events_ctx,
        "subjects": subjects_summary,
        "total_duration_sec": round(total_duration, 2),
    }
    _ph("study_created", user["sub"], {"study_id": s_id, "study_name": req.name, "modality": req.modality, "n_channels": subjects[0]["info"].n_channels, "n_subjects": len(subjects), "sfreq": req.sfreq, "has_montage": bool(req.montage), "has_events": bool(req.events)})
    return {"study_id": s_id, "name": req.name}

@app.post("/api/study/load/{study_id}")
async def load_study(study_id: int, user: dict = Depends(get_current_user)):
    uid = int(user["sub"]); session = get_user_session(uid)
    db = SessionLocal()
    study = db.query(Study).filter(Study.id == study_id, Study.user_id == uid).first()
    if not study:
        db.close()
        raise HTTPException(status_code=404, detail="Study not found")
    
    # Restore legacy subjects from files_json
    session["subjects"] = []
    if study.files_json:
        try:
            meta_list = json.loads(study.files_json)
            for i, meta in enumerate(meta_list):
                path = meta.get("path")
                if path and os.path.exists(path):
                    raw, info = eeg_loader.load(path)
                    if raw:
                        # Rebuild preview
                        n_samples = int(min(5.0, info.duration_sec) * info.sfreq)
                        data = raw.get_data(stop=n_samples)
                        times = np.linspace(0, n_samples/info.sfreq, n_samples).tolist()

                        # Demean and use robust per-channel std (same as upload)
                        for j in range(data.shape[0]):
                            data[j] -= np.mean(data[j])
                        stds = np.std(data, axis=1)
                        std_val = float(np.median(stds)) if len(stds) > 0 and np.median(stds) > 0 else 1e-6
                        if std_val < 1e-8: std_val = 1e-6

                        n_ch = info.n_channels
                        ch_pre = []
                        for j in range(n_ch):
                            offset = float((n_ch - 1 - j) * std_val * 6)
                            ch_pre.append({"name": info.channel_names[j], "data": [float(v) + offset for v in data[j].tolist()]})
                        
                        session["subjects"].append({
                            "id": i, "name": meta["name"], "path": path, "raw": raw, "info": info, 
                            "preview": {"times": times, "channels": ch_pre}
                        })
        except Exception as e:
            logger.error(f"Failed to restore subjects: {e}")

    # Build per-subject summary
    subjects_summary = []
    total_duration = 0.0
    for s in session["subjects"]:
        info = s["info"]
        total_duration += info.duration_sec
        subjects_summary.append({
            "name": s["name"],
            "n_channels": info.n_channels,
            "channel_names": info.channel_names,
            "sfreq": info.sfreq,
            "duration_sec": round(info.duration_sec, 2),
            "format": info.format_name,
        })

    session["study_id"] = study.id
    session["study_ctx"] = {
        "study_id": study.id,
        "name": study.name,
        "modality": study.modality,
        "n_subjects": len(session["subjects"]),
        "sfreq": study.sfreq,
        "conditions": study.conditions,
        "reference": study.reference,
        "notes": study.notes,
        "montage": [],
        "subjects": subjects_summary,
        "total_duration_sec": round(total_duration, 2),
    }

    if session["subjects"]:
        first = session["subjects"][0]["info"]
        session["study_ctx"].update({
            "n_channels": first.n_channels,
            "channel_names": first.channel_names,
            "duration_sec": round(first.duration_sec, 2),
            "file_format": first.format_name,
        })
    db.close()
    return {"message": "Study loaded", "study": session["study_ctx"]}

@app.delete("/api/study/subject/{subject_id}/channel/{channel_idx}")
async def delete_channel_endpoint(subject_id: int, channel_idx: int, user: dict = Depends(get_current_user)):
    uid = int(user["sub"]); session = get_user_session(uid)
    subjects = session.get("subjects", [])
    subj = next((s for s in subjects if s["id"] == subject_id), None)
    if not subj: raise HTTPException(status_code=404, detail="Subject not found")
    
    raw = subj.get("raw")
    if raw is None: raise HTTPException(status_code=400, detail="Data not loaded")
    
    if channel_idx < 0 or channel_idx >= len(raw.ch_names):
        raise HTTPException(status_code=400, detail="Invalid channel index")
        
    ch_name = raw.ch_names[channel_idx]
    try:
        raw.drop_channels([ch_name])
        # Update cached info
        subj["info"].n_channels = len(raw.ch_names)
        subj["info"].channel_names = list(raw.ch_names)
        # Update preview (optional but keeps things consistent)
        subj["preview"]["channels"] = [c for c in subj["preview"]["channels"] if c["name"] != ch_name]
        
        # If this was the active study, update study_ctx too
        if session.get("study_ctx") and session["study_ctx"].get("subjects"):
            for s_meta in session["study_ctx"]["subjects"]:
                if s_meta["name"] == subj["name"]:
                    s_meta["n_channels"] = len(raw.ch_names)
                    s_meta["channel_names"] = list(raw.ch_names)
        
        _ph("channel_deleted", user["sub"], {"channel_name": ch_name, "remaining": len(raw.ch_names)})
        return {"message": f"Dropped channel {ch_name}", "n_channels": len(raw.ch_names)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/study/subject/{subject_id}")
async def delete_subject(subject_id: int, user: dict = Depends(get_current_user)):
    uid = int(user["sub"]); session = get_user_session(uid)
    subjects = session.get("subjects", [])
    # Find and remove
    session["subjects"] = [s for s in subjects if s["id"] != subject_id]
    return {"message": f"Subject {subject_id} deleted"}

@app.delete("/api/study/subjects/clear")
async def clear_subjects(user: dict = Depends(get_current_user)):
    uid = int(user["sub"]); session = get_user_session(uid)
    session["subjects"] = []
    session.pop("study_ctx", None)
    session.pop("study_id", None)
    return {"message": "All subjects cleared"}

@app.delete("/api/study/delete/{study_id}")
async def delete_study(study_id: int, user: dict = Depends(get_current_user)):
    uid = int(user["sub"]); session = get_user_session(uid)
    db = SessionLocal()
    try:
        study = db.query(Study).filter(Study.id == study_id, Study.user_id == uid).first()
        if not study:
            raise HTTPException(status_code=404, detail="Study not found")
        # Delete associated chat messages
        db.query(ChatMessage).filter(ChatMessage.study_id == study_id, ChatMessage.user_id == uid).delete()
        # Delete associated analysis jobs
        db.query(AnalysisJob).filter(AnalysisJob.study_id == study_id).delete()
        # Delete the study
        db.delete(study)
        db.commit()
        # Clear session if this was the active study
        if session.get("study_id") == study_id:
            session["subjects"] = []
            session.pop("study_ctx", None)
            session.pop("study_id", None)
            session["chat_history"] = []
        return {"message": "Study deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _ph("study_deleted", user["sub"], {"study_id": study_id})
        db.close()

def _build_chat_ctx(session: dict) -> dict:
    """Build an enriched context dict from the current session (data stats + pipeline results)."""
    ctx = dict(session.get("study_ctx", {}))
    subjects = session.get("subjects", [])
    montage  = ctx.get("montage", [])
    if subjects:
        data_stats = []
        for s in subjects:
            raw = s.get("raw")
            if raw is not None:
                try:
                    cap_samples = min(int(raw.n_times), int(60 * raw.info["sfreq"]))
                    data = raw.get_data(stop=cap_samples)
                    info = s["info"]
                    channel_stats = []
                    for ch_i in range(min(info.n_channels, data.shape[0])):
                        ch_data = data[ch_i]
                        electrode = montage[ch_i] if ch_i < len(montage) and montage[ch_i] else info.channel_names[ch_i]
                        channel_stats.append({
                            "electrode": electrode,
                            "raw_name": info.channel_names[ch_i],
                            "amplitude_uV_range": [round(float(np.min(ch_data)) * 1e6, 2), round(float(np.max(ch_data)) * 1e6, 2)],
                            "amplitude_uV_std": round(float(np.std(ch_data)) * 1e6, 2),
                        })
                    data_stats.append({
                        "name": s["name"],
                        "n_samples": int(raw.n_times),
                        "amplitude_uV_range": [round(float(np.min(data)) * 1e6, 2), round(float(np.max(data)) * 1e6, 2)],
                        "amplitude_uV_std": round(float(np.std(data)) * 1e6, 2),
                        "channel_stats": channel_stats,
                    })
                except Exception:
                    pass
        if data_stats:
            ctx["data_stats"] = data_stats
    pipeline = session.get("pipeline")
    if pipeline is not None:
        try:
            pipeline_summary = pipeline.get_results_summary()
            if pipeline_summary.get("band_powers") or pipeline_summary.get("erp_peak"):
                ctx["pipeline_results"] = {
                    "band_powers": pipeline_summary.get("band_powers", {}),
                    "erp_peak": pipeline_summary.get("erp_peak", {}),
                    "status": pipeline_summary.get("status", "idle"),
                }
        except Exception:
            pass
    return ctx


def _save_chat_turn(session: dict, uid: int, user_msg: str,
                    response: dict, model: str) -> None:
    """Append a user/assistant turn to session history and persist to DB."""
    assistant_content = response.get("message", response.get("understanding", ""))
    session["chat_history"].append({"role": "user",      "content": user_msg})
    session["chat_history"].append({"role": "assistant", "content": assistant_content})

    if response.get("type") == "pipeline":
        steps = response.get("pipeline_steps", [])
        if steps and session.get("subjects"):
            session["pending_steps"] = steps
        elif steps and not session.get("subjects"):
            response["message"] = (response.get("message", "") +
                                   " (Please upload EEG files first so I can run this pipeline.)").strip()
            session["pending_steps"] = []

    study_id = session.get("study_id")
    if study_id:
        db = SessionLocal()
        try:
            db.add(ChatMessage(study_id=study_id, user_id=uid, role="user",      content=user_msg,          model=model))
            db.add(ChatMessage(study_id=study_id, user_id=uid, role="assistant", content=assistant_content, model=model))
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()


@app.post("/api/ai/chat")
async def ai_chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    uid = int(user["sub"]); session = get_user_session(uid)
    ctx = _build_chat_ctx(session)
    response = ai_interpreter.interpret(req.message, ctx, session["chat_history"], model=req.model)
    _save_chat_turn(session, uid, req.message, response, req.model)
    if response.get("type") == "pipeline" and session.get("pending_steps"):
        response["pipeline_steps"] = session["pending_steps"]
    _ph("ai_chat_sent", user["sub"], {
        "message_length": len(req.message),
        "model": req.model,
        "response_type": response.get("type", "unknown"),
        "has_pipeline_steps": bool(response.get("pipeline_steps")),
        "n_pipeline_steps": len(response.get("pipeline_steps", [])),
    })
    return response


@app.post("/api/ai/chat/stream")
async def ai_chat_stream(req: ChatRequest, user: dict = Depends(get_current_user)):
    """SSE streaming version of /api/ai/chat.
    Events:
      data: {"type":"chunk","text":"<partial text>"}
      data: {"type":"done","response":{...full response dict...}}
      data: {"type":"error","message":"<error>"}
    """
    uid = int(user["sub"]); session = get_user_session(uid)
    ctx      = _build_chat_ctx(session)
    history  = list(session.get("chat_history", []))  # snapshot for thread safety

    def sync_generate():
        final_response = None
        try:
            for event in ai_interpreter.interpret_stream(
                req.message, ctx, history, model=req.model
            ):
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "done":
                    final_response = event.get("response", {})
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        # Persist turn after streaming completes
        if final_response:
            _save_chat_turn(session, uid, req.message, final_response, req.model)
            if final_response.get("type") == "pipeline" and session.get("pending_steps"):
                # Emit an extra event so the frontend gets the saved steps
                final_response["pipeline_steps"] = session["pending_steps"]
                yield f"data: {json.dumps({'type': 'pipeline_steps_updated', 'steps': session['pending_steps']})}\n\n"

    return StreamingResponse(
        iterate_in_threadpool(sync_generate()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/ai/chat/history")
async def get_chat_history(user: dict = Depends(get_current_user)):
    uid = int(user["sub"]); session = get_user_session(uid)
    study_id = session.get("study_id")
    if not study_id:
        return {"messages": [], "pending_steps": []}
    db = SessionLocal()
    try:
        msgs = (
            db.query(ChatMessage)
            .filter(ChatMessage.study_id == study_id, ChatMessage.user_id == uid)
            .order_by(ChatMessage.created_at)
            .all()
        )
        result = [{"role": m.role, "content": m.content, "model": m.model} for m in msgs]
        session["chat_history"] = result
        return {"messages": result, "pending_steps": session.get("pending_steps", [])}
    finally:
        db.close()


@app.post("/api/ai/chat/clear")
async def clear_chat_history(user: dict = Depends(get_current_user)):
    uid = int(user["sub"]); session = get_user_session(uid)
    study_id = session.get("study_id")
    if not study_id:
        return {"message": "No study active"}
    db = SessionLocal()
    try:
        db.query(ChatMessage).filter(ChatMessage.study_id == study_id, ChatMessage.user_id == uid).delete()
        db.commit()
        session["chat_history"] = []
        return {"message": "Chat history cleared"}
    finally:
        db.close()


@app.post("/api/pipeline/run")
async def pipeline_run(user: dict = Depends(get_current_user)):
    uid = int(user["sub"]); session = get_user_session(uid)
    subjects = session.get("subjects")
    steps = session.get("pending_steps")
    
    if not subjects:
        raise HTTPException(status_code=400, detail="No EEG data loaded. Please upload files and create a study first.")
    if not steps:
        raise HTTPException(status_code=400, detail="No pipeline steps available. Ask the AI assistant to generate a pipeline first.")
    
    pipeline = get_user_pipeline(uid)
    pipeline.reset()
    raw = subjects[0].get("raw")
    if raw is None:
        raise HTTPException(status_code=400, detail="Raw data not available. Please re-upload the EEG file.")
    raw = raw.copy()

    # Apply user-selected montage from study context
    montage_names = session.get("study_ctx", {}).get("montage", [])
    if montage_names:
        try:
            import mne

            # Step 1: If user mapped channel slots to electrode names, rename raw channels
            ch_names = list(raw.ch_names)
            rename_map = {}
            for i, mont_name in enumerate(montage_names):
                if i < len(ch_names) and mont_name and mont_name.strip():
                    new_name = mont_name.strip()
                    if ch_names[i] != new_name:
                        rename_map[ch_names[i]] = new_name
            if rename_map:
                raw.rename_channels(rename_map)
                print(f"[MONTAGE] Renamed {len(rename_map)} channels: {rename_map}", flush=True)

            # Step 2: Try to apply a standard montage to get electrode positions
            for montage_name in ["standard_1005", "standard_1020", "standard_1010"]:
                try:
                    std_montage = mne.channels.make_standard_montage(montage_name)
                    raw.set_montage(std_montage, on_missing="warn", match_case=True)
                    n_with_pos = sum(1 for ch in raw.info["chs"] if ch.get("loc") is not None and any(ch["loc"][:3]))
                    print(f"[MONTAGE] Applied {montage_name}: {n_with_pos}/{len(raw.ch_names)} channels have positions", flush=True)
                    break
                except Exception as e:
                    print(f"[MONTAGE] {montage_name} failed: {e}", flush=True)
                    continue
        except Exception as e:
            print(f"[MONTAGE] Warning: Could not apply montage: {e}", flush=True)

    pipeline.run(steps, raw)
    _ph("pipeline_started", user["sub"], {"n_steps": len(steps), "step_names": [s.get("name", "") for s in steps]})
    return {"status": "started", "n_steps": len(steps)}


@app.get("/api/pipeline/status")
async def pipeline_status(user: dict = Depends(get_current_user)):
    uid = int(user["sub"])
    pipeline = get_user_pipeline(uid)
    results = pipeline.get_results_summary()
    return _sanitize({
        "status": pipeline.status, "current_step": pipeline.current_step, "total_steps": len(pipeline.steps),
        "steps": [{"index": i, "name": s.get("name", "Step"), "step_status": results["steps"].get(i, {}).get("status", "pending")} for i, s in enumerate(pipeline.steps)],
        "log": results.get("log", []),
        "band_powers": results.get("band_powers", {}), "figures": {k: os.path.basename(v) for k, v in results.get("figures", {}).items()},
    })

@app.post("/api/pipeline/pause")
async def pipeline_pause(user: dict = Depends(get_current_user)):
    uid = int(user["sub"])
    get_user_pipeline(uid).pause()
    _ph("pipeline_paused", user["sub"])
    return {"status": "paused"}

@app.post("/api/pipeline/resume")
async def pipeline_resume(user: dict = Depends(get_current_user)):
    uid = int(user["sub"])
    get_user_pipeline(uid).resume()
    _ph("pipeline_resumed", user["sub"])
    return {"status": "running"}

@app.post("/api/pipeline/stop")
async def pipeline_stop(user: dict = Depends(get_current_user)):
    uid = int(user["sub"])
    get_user_pipeline(uid).stop()
    _ph("pipeline_stopped", user["sub"])
    return {"status": "stopped"}

@app.get("/api/study/download/{subject_id}")
async def download_subject(subject_id: int, user: dict = Depends(get_current_user)):
    uid = int(user["sub"]); session = get_user_session(uid)
    subjects = session.get("subjects", [])
    subject = next((s for s in subjects if s["id"] == subject_id), None)
    if not subject: raise HTTPException(status_code=404, detail="Subject not found.")
    
    path = subject["path"]
    if not os.path.exists(path): raise HTTPException(status_code=404, detail="File not found.")
    ext = os.path.splitext(path)[1].lower()
    media_types = {
        ".edf": "application/octet-stream",
        ".bdf": "application/octet-stream",
        ".fif": "application/octet-stream",
        ".set": "application/octet-stream",
        ".vhdr": "application/octet-stream",
        ".csv": "text/csv",
        ".tsv": "text/tab-separated-values",
        ".npy": "application/octet-stream",
        ".fdt": "application/octet-stream",
        ".xdf": "application/octet-stream",
    }
    media_type = media_types.get(ext, "application/octet-stream")
    return FileResponse(path, filename=os.path.basename(path), media_type=media_type)


@app.post("/api/results/generate")
async def generate_results(user: dict = Depends(get_current_user)):
    uid = int(user["sub"]); session = get_user_session(uid)
    subjects = session.get("subjects", [])
    if not subjects:
        raise HTTPException(status_code=400, detail="No subjects loaded. Please upload EEG data first.")
    
    pipeline = get_user_pipeline(uid)
    # Check if pipeline has any results (band powers, erp peaks, or completed steps)
    has_results = (
        pipeline.results.get("band_powers") or 
        pipeline.results.get("erp_peak") or 
        pipeline.step_outputs
    )
    if not has_results:
        raise HTTPException(status_code=400, detail="No analysis results available. Please run the pipeline first.")
    
    info = subjects[0]["info"]; ctx = session.get("study_ctx", {})
    results = pipeline.get_results_summary()
    statistics = results.get("statistics", {})
    study_info = {
        "name": ctx.get("name", "CortexIQ Study"),
        "modality": ctx.get("modality", "EEG"),
        "n_channels": info.n_channels,
        "sfreq": info.sfreq,
        "conditions": ctx.get("conditions", ""),
        "duration_sec": info.duration_sec,
        "subject_count": ctx.get("n_subjects", len(subjects)),
    }

    # Use local interpretation (no AI calls to avoid image errors)
    interp = _build_local_interpretation(statistics, study_info)
    methods = _build_local_methods(pipeline.steps, study_info)

    # Generate report files
    try:
        zip_p, pdf_p, csv_p = eeg_reporter.generate(study_info, results, pipeline.figures, interp, methods)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")
    
    band_powers = results.get("band_powers", {})
    erp_peak = results.get("erp_peak", {})
    figures = {k: os.path.basename(v) for k, v in pipeline.figures.items() if v and os.path.exists(v)}
    
    step_summaries = []
    for step_id, step_data in results.get("steps", {}).items():
        if isinstance(step_data, dict):
            step_summaries.append({
                "step": step_id,
                "name": step_data.get("name", f"Step {step_id}"),
                "status": step_data.get("status", "unknown"),
                "summary": step_data.get("summary", ""),
            })
    
    result = _sanitize({
        "interpretation": interp,
        "methods": methods,
        "band_powers": band_powers,
        "erp_peak": erp_peak,
        "figures": figures,
        "step_summaries": step_summaries,
        "statistics": statistics,
        "files": {
            "pdf": os.path.basename(pdf_p),
            "zip": os.path.basename(zip_p),
            "csv": os.path.basename(csv_p),
        },
        "study_info": study_info,
    })
    _ph("results_generated", user["sub"], {
        "study_name": study_info.get("name", ""),
        "n_channels": study_info.get("n_channels", 0),
        "n_figures": len(figures),
        "has_band_powers": bool(band_powers),
        "has_erp": bool(erp_peak),
        "n_steps": len(step_summaries),
    })
    return result


def _build_local_interpretation(statistics: dict, study_info: dict) -> str:
    """Build an APA 7 style interpretation from computed statistics."""
    parts = []
    g = statistics.get("descriptive", {}).get("global", {})
    if g:
        parts.append(
            f"The present analysis examined a {g.get('n_channels', 'N/A')}-channel "
            f"{study_info.get('modality', 'EEG')} recording "
            f"(duration = {g.get('duration_sec', 'N/A')} s, sampling rate = {g.get('sampling_rate_Hz', 'N/A')} Hz). "
            f"The global signal amplitude was M = {g.get('global_mean_uV', 0):.2f} µV "
            f"(SD = {g.get('global_std_uV', 0):.2f} µV), "
            f"ranging from {g.get('global_min_uV', 0):.2f} to {g.get('global_max_uV', 0):.2f} µV."
        )

    # Cross-channel correlation
    corr = statistics.get("descriptive", {}).get("cross_channel_correlation", {})
    if corr and corr.get("mean_r") is not None:
        parts.append(
            f"Cross-channel coherence was assessed via pairwise Pearson correlations, yielding a mean r = {corr['mean_r']:.4f} "
            f"(range: {corr.get('min_r', 0):.4f} to {corr.get('max_r', 0):.4f}), "
            f"suggesting {'moderate to high' if abs(corr['mean_r']) > 0.3 else 'low'} inter-channel linear dependence."
        )

    bp = statistics.get("band_analysis", {})
    if bp and "total_power_V2_Hz" in bp:
        bands_sorted = sorted(
            [(k, v) for k, v in bp.items() if isinstance(v, dict) and "relative_power_pct" in v],
            key=lambda x: x[1]["relative_power_pct"], reverse=True
        )
        if bands_sorted:
            dominant = bands_sorted[0]
            parts.append(
                f"Spectral analysis using Welch's method revealed that the {dominant[0]} band was dominant, "
                f"accounting for {dominant[1]['relative_power_pct']:.1f}% of total spectral power. "
                f"The relative power distribution across bands was as follows: "
                + ", ".join([f"{k} = {v['relative_power_pct']:.1f}%" for k, v in bands_sorted])
                + "."
            )

    erp = statistics.get("erp_analysis", {})
    if erp and erp.get("peak_amplitude_uV"):
        parts.append(
            f"Event-related potential analysis identified a peak amplitude of "
            f"{erp['peak_amplitude_uV']:.2f} µV at a latency of {erp['peak_latency_ms']:.1f} ms, "
            f"observed at electrode {erp['peak_channel']}."
        )

    ep = statistics.get("epoch_analysis", {})
    if ep:
        parts.append(
            f"The continuous data were segmented into {ep.get('n_epochs', 'N/A')} epochs "
            f"of {ep.get('epoch_duration_sec', 'N/A')} s duration each, "
            f"yielding a total analyzed epoch time of {ep.get('total_epoch_time_sec', 'N/A')} s."
        )

    return " ".join(parts) if parts else "No analysis results available. Please execute the processing pipeline first."


def _build_local_methods(pipeline_steps: list, study_info: dict) -> str:
    """Build an APA 7 style methods paragraph."""
    if not pipeline_steps:
        return ""
    step_names = [s.get("name", "Unknown step") for s in pipeline_steps if isinstance(s, dict)]
    n_subj = study_info.get('subject_count', 1)
    n_ch = study_info.get('n_channels', 'N/A')
    sfreq = study_info.get('sfreq', 'N/A')
    return (
        f"Electroencephalographic (EEG) data were recorded from {n_ch} scalp electrodes "
        f"at a sampling rate of {sfreq} Hz (N = {n_subj}). "
        f"All signal processing was performed using NeuraGentLab's CortexIQ platform "
        f"(built on MNE-Python; Gramfort et al., 2013). "
        f"The following preprocessing and analysis pipeline was applied sequentially: "
        f"{'; '.join(step_names)}. "
        f"Spectral power was estimated using Welch's method. "
        f"All statistical computations were performed on the preprocessed data."
    )

@app.get("/api/results/download/{filename}")
async def download_result(filename: str, user: dict = Depends(get_current_user)):
    _ph("result_downloaded", user["sub"], {"filename": filename, "file_type": os.path.splitext(filename)[1]})
    safe_name = os.path.basename(filename)
    path = os.path.join(RESULTS_DIR, safe_name)
    if not os.path.exists(path): raise HTTPException(status_code=404)
    # Set proper content type based on extension
    ext = os.path.splitext(safe_name)[1].lower()
    media_types = {
        ".pdf": "application/pdf",
        ".csv": "text/csv",
        ".zip": "application/zip",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
    }
    media_type = media_types.get(ext, "application/octet-stream")
    return FileResponse(path, filename=safe_name, media_type=media_type)

@app.get("/api/analytics/config")
async def analytics_config():
    """Return PostHog configuration for frontend SDK. Public endpoint."""
    return {"posthog_api_key": POSTHOG_API_KEY, "posthog_host": POSTHOG_HOST}

@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    with open(os.path.join(os.path.dirname(__file__), "static", "demo", "index.html"), "r") as f: return HTMLResponse(content=f.read())

@app.get("/investor", response_class=HTMLResponse)
async def investor_page():
    with open(os.path.join(os.path.dirname(__file__), "static", "investor", "index.html"), "r") as f: return HTMLResponse(content=f.read())

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(os.path.dirname(__file__), "static", "index.html"), "r") as f: return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
