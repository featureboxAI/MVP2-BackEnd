from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from zipfile import ZipFile, BadZipFile
import os
import tempfile
from google.cloud import storage
import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
from typing import Dict, List
import traceback   ## TRACEBACK LOGGING
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel 


# Set Google Cloud credentials using relative path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
credentials_path = os.path.join(current_dir, "service_account_key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Configuration constants
ITEM_COL_F = "Item"
SEASONAL_P = 12
HIST_END = pd.Timestamp("2025-05-01")
FORECAST_START = pd.Timestamp("2025-06-01")


app = FastAPI()

origins = [
    "https://45006fa9-15b6-4438-a9a4-94de64110d9d.lovableproject.com",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=(
        r"(https:\/\/.*\.lovable\.(app|com))"   # any Lovable preview
        r"|^https:\/\/[a-zA-Z0-9\-]+\.lovableproject\.com$"
        r"|(^http:\/\/localhost(:\d+)?$)"       # localhost with optional :port
        r"|(^http:\/\/127\.0\.0\.1(:\d+)?$)"    # 127.0.0.1 with optional :port
    ),
    # Will match:
    #   https://preview--featurebox-ai-ui-mvp-2-03.lovable.app
    #   https://45006fa9-....lovableproject.com
    allow_credentials=False,   # keep False with a wildcard / regex
    allow_methods=["*"],    # or ["*"] if you’ll add GET etc.
    allow_headers=["*"],
)

# Import forecast logic
from app.forecasting import generate_forecasts

def upload_to_gcs(file_path: str) -> None:
    try:
        client = storage.Client()
        print(f"Using project: {client.project}")
        print(f"Authenticated as: {client._credentials.service_account_email}")
        bucket = client.bucket("featurebox-ai-uploads")
        blob_name = os.path.basename(file_path)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        print(f"Successfully uploaded {blob_name} to GCS")
    except Exception as e:
        print(f"Upload error: {str(e)}")  # ### TRACEBACK LOGGING
        raise HTTPException(status_code=500, detail=f"GCS Upload failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "FastAPI backend is running!"}


@app.post("/upload/")
async def upload_zip(file: UploadFile = File(...)):
    print(" Received type:", type(file))
    try:
        if not file.filename.endswith(".zip"):
            raise HTTPException(status_code=400, detail="Only ZIP files are allowed.")

        tmpdir = tempfile.mkdtemp(prefix="zip_extract_")
        zip_path = os.path.join(tmpdir, file.filename)

        try:
            file.file.seek(0)
            with open(zip_path, "wb") as buffer:
                while chunk := file.file.read(1024 * 1024):
                    buffer.write(chunk)

            with ZipFile(zip_path) as z:
                if z.testzip() is not None:
                    raise HTTPException(status_code=400, detail="Corrupt ZIP file detected.")
                z.extractall(tmpdir)

            extracted_files = [
                os.path.join(root, f)
                for root, _, files in os.walk(tmpdir)
                for f in files
                if f.endswith(".xlsx") and "__MACOSX" not in root
            ]

            if not extracted_files:
                raise HTTPException(status_code=400, detail="No Excel (.xlsx) files found in ZIP.")

            uploaded_files = []
            forecast_file = None

            for excel_file in extracted_files:
                print(f"🔍 Checking file: {os.path.basename(excel_file)}")
                upload_to_gcs(excel_file)
                uploaded_files.append(os.path.basename(excel_file))
                if os.path.basename(excel_file).lower() == "sprouts_data.xlsx":
                    forecast_file = excel_file

            if not forecast_file:
                return JSONResponse(content={
                    "status": "success",
                    "message": "Files uploaded but sprouts_data.xlsx not found",
                    "uploaded_files": uploaded_files
                })

            result = generate_forecasts(forecast_file)

            if result['status'] == 'error':
                raise HTTPException(status_code=500, detail=result['message'])

            return FileResponse(
                path=result['forecast_file'],
                filename=os.path.basename(result['forecast_file']),
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

        except BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid or corrupt ZIP file.")
        except Exception as e:
            print(" Exception during ZIP processing:\n", traceback.format_exc())  # ### TRACEBACK LOGGING
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

        finally:
            if os.path.exists(tmpdir):
                try:
                    import shutil
                    shutil.rmtree(tmpdir)
                except Exception:
                    pass

    except Exception as e:
        print(" Outer exception:\n", traceback.format_exc())  # ### TRACEBACK LOGGING
        return JSONResponse(
            status_code=422,
            content={
                "detail": [{
                    "type": "value_error",
                    "loc": ["body", "file"],
                    "msg": "Expected UploadFile, got str",
                    "input": "string",
                    "ctx": {"error": {}}
                }]
            }
        )


class LoginRequest(BaseModel):
    email: str
    password: str

# Dummy users for now (can replace later with DB logic)
DUMMY_USERS = {
    "user1@customer1.com": {
        "password": "pass123",
        "tenant": "customer1",
        "backend_url": "https://featurebox-ai-service-666676702816.us-west1.run.app"
    },
    "user2@customer2.com": {
        "password": "pass456",
        "tenant": "customer2",
        "backend_url": "https://backend.customer2.com"
    },
}

@app.post("/login")
async def login_user(data: LoginRequest):
    user = DUMMY_USERS.get(data.email)

    if not user or user["password"] != data.password:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Fake token for now
    return {
        "status": "success",
        "token": "fake.jwt.token.for." + data.email,
        "tenant": user["tenant"],
        "backend_url": user["backend_url"]
    }