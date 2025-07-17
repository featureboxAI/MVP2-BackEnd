from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from zipfile import ZipFile, BadZipFile
import os
import requests
import tempfile
import shutil
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

app = FastAPI()

origins = [
    "https://lovable.dev/projects/0189a77d-15e6-4282-bb19-cb7fdc4d7803",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=(
        r"(https:\/\/.*\.lovable\.(app|com))"   # any Lovable preview
        r"|^https:\/\/[a-zA-Z0-9\-]+\.lovableproject\.com$"
        r"|(^http:\/\/localhost(:\d+)?$)"       # localhost with optional :port
        r"|(^http:\/\/127\.0\.0\.1(:\d+)?$)"    # 127.0.0.1 with optional :port
        r"|(^http:\/\/10\.0\.0\.122(:\d+)?$)"
    ),

    allow_credentials=False,   # keep False with a wildcard / regex
    allow_methods=["*"],    # or ["*"] if you’ll add GET etc.
    allow_headers=["*"],
)


# Import forecast logic
from .forecasting import generate_forecasts

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

def trigger_forecast_on_vm(core_gcs, cons_gcs=None, az_gcs=None):
    vm_url = "http://34.63.139.152:8000/run-forecast" 
    payload = {
        "core_path": core_gcs,
        "cons_path": cons_gcs,
        "az_path": az_gcs
    }
    # Remove None values
    clean_payload = {k: v for k, v in payload.items() if v}
    response = requests.post(vm_url, json=clean_payload)
    return response.json()


@app.get("/")
async def root():
    return {"message": "FastAPI backend is running!"}


@app.post("/upload/")
async def upload_zip(file: UploadFile = File(...)):
    print(" [BACKEND] Received ZIP upload:", type(file))
    
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
            cons_file = None
            az_file = None

            for excel_file in extracted_files:
                fname = os.path.basename(excel_file).lower()
                print(f" [BACKEND] Found Excel file: {fname}")
                print(f" Checking file: {fname}")
                upload_to_gcs(excel_file)
                uploaded_files.append(os.path.basename(excel_file))
                
                # File classification logic
                if "amazon" in fname:
                    az_file = excel_file
                elif any(kw in fname for kw in ["cons", "spins", "demand"]):
                    cons_file = excel_file
                elif forecast_file is None:
                    forecast_file = excel_file  # fallback to first remaining Excel file

            if not forecast_file:
                return JSONResponse(content={
                    "status": "success",
                    "message": "Files uploaded but sprouts_data.xlsx not found",
                    "uploaded_files": uploaded_files
                })
            
             # DEBUG: Print what we're sending to VM
            print(" [BACKEND] Triggering forecast on VM with:")
            print(" - Forecast file:", forecast_file)
            print(" - Cons file:", cons_file)
            print(" - Amazon file:", az_file)

            # Upload all to GCS
            gcs_core = upload_to_gcs(forecast_file, "featurebox-ai-uploads")
            gcs_cons = upload_to_gcs(cons_file, "featurebox-ai-uploads") if cons_file else None
            gcs_az   = upload_to_gcs(az_file, "featurebox-ai-uploads") if az_file else None

            # Call VM with JSON body of GCS paths
            vm_result = trigger_forecast_on_vm(gcs_core, gcs_cons, gcs_az)

            # Log response
            print("[BACKEND] VM responded with:", vm_result)

            return JSONResponse(content={
                "status": "success",
                "message": vm_result.get("message"),
                "metrics_shape": vm_result.get("metrics_shape"),
                "forecast_shape": vm_result.get("forecast_shape"),
                "raw_response": vm_result
            })
        

        except BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid or corrupt ZIP file.")
        except Exception as e:
            print(" Exception during ZIP processing:\n", traceback.format_exc())  # ### TRACEBACK LOGGING
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

        finally:
            if os.path.exists(tmpdir):
                try:
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