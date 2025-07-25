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
from fastapi.responses import FileResponse
from datetime import timedelta
from fastapi.responses import StreamingResponse
import requests
from io import BytesIO


# Set Google Cloud credentials using relative path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
credentials_path = os.path.join(current_dir, "service_account_key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

app = FastAPI()

latest_forecast = {}

origins = [
    "https://featurebox-ai-ui-mvp-2-05.lovable.app/",
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

def generate_signed_url(bucket_name, blob_name, expiration_minutes=10):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(minutes=expiration_minutes),
        method="GET",
    )
    return url

def upload_to_gcs(file_path: str, bucket_name = "featurebox-ai-uploads") -> str:
    try:
        client = storage.Client()
        print(f"Using project: {client.project}")
        print(f"Authenticated as: {client._credentials.service_account_email}")
        bucket = client.bucket("featurebox-ai-uploads")
        blob_name = f"excel_uploads/{os.path.basename(file_path)}"  # Upload inside subfolder
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)

        print(f"Successfully uploaded {blob_name} to GCS")

        return blob_name  

    except Exception as e:
        print(f"Upload error: {str(e)}")  # ### TRACEBACK LOGGING
        raise HTTPException(status_code=500, detail=f"GCS Upload failed: {str(e)}")
    

def trigger_forecast_on_vm(core_gcs, cons_gcs=None, az_gcs=None):
    if not core_gcs:
        raise ValueError("core_gcs must be provided to trigger_forecast_on_vm")

    vm_url = "http://35.223.133.115:8000/run-forecast" 
    # vm_url = "http://127.0.0.1:8002/run-forecast"

    payload = {
        "core_path": core_gcs,
        "cons_path": cons_gcs,
        "az_path": az_gcs
    }
    clean_payload = {k: v for k, v in payload.items() if v}
    
    print("[DEBUG] Payload being sent to VM:", clean_payload)

    # response = requests.post(vm_url, json=clean_payload)

    try:
        response = requests.post(vm_url, json=clean_payload, timeout=60)
        # latest_forecast["gcs_path"] = response.get("forecast_gcs")
        latest_forecast["gcs_path"] = response.json().get("forecast_gcs")  # NEW: store path from VM


        print("[DEBUG] VM response status:", response.status_code)
        print("[DEBUG] VM response text:", response.text)

        # safer return block
        try:
            return response.json()
        except Exception as json_err:
            print("[ERROR] Could not parse JSON from VM:", json_err)
            return {"status": "unknown", "raw": response.text}

    except Exception as e:
        print("[ERROR] VM call failed:", e)
        raise HTTPException(status_code=500, detail=f"VM request failed: {str(e)}")


@app.get("/")
async def root():
    return {"message": "FastAPI backend is running!"}


@app.post("/upload/")
async def upload_zip(file: UploadFile = File(...)):
    print("Received type:", type(file))
    print("Filename:", file.filename)
    print("ContentType:", file.content_type)



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
                
                if "amazon" in fname or "az" in fname:
                    az_file = excel_file
                elif any(kw in fname for kw in ["cons", "spins", "demand"]):
                    cons_file = excel_file
                elif forecast_file is None or any(kw in fname for kw in ["core", "allsku", "sprout"]):  # ✅ better fallback
                    forecast_file = excel_file

            if not forecast_file:
                raise HTTPException(status_code=400, detail="Could not identify forecast file (core/allsku/sprout) from uploaded ZIP.")

            
             # DEBUG: Print what we're sending to VM
            print(" [BACKEND] Triggering forecast on VM with:")
            print(" - Forecast file:", forecast_file)
            print(" - Cons file:", cons_file)
            print(" - Amazon file:", az_file)

            print(" [DEBUG] Selected forecast_file:", forecast_file)

            # Upload all to GCS
            BUCKET_NAME = "featurebox-ai-uploads" 
            gcs_core = upload_to_gcs(forecast_file, "featurebox-ai-uploads")
            print(" [DEBUG] Uploaded forecast file to GCS as:", gcs_core)

            gcs_cons = upload_to_gcs(cons_file, "featurebox-ai-uploads") if cons_file else None
            gcs_az   = upload_to_gcs(az_file, "featurebox-ai-uploads") if az_file else None

            # Call VM with JSON body of GCS paths
            vm_result = trigger_forecast_on_vm(gcs_core, gcs_cons, gcs_az)
            print("[DEBUG] Result from VM:", vm_result)

            print("[DEBUG] VM returned:", vm_result)

            # Log response
            print("[BACKEND] VM responded with:", vm_result)

            #  Generate signed download URL from forecast_gcs if returned
            if vm_result.get("status") == "success" and "forecast_gcs" in vm_result:
                gcs_path = vm_result["forecast_gcs"]
                blob_name = gcs_path.replace(f"gs://{BUCKET_NAME}/", "")  #  Extract blob path
                signed_url = generate_signed_url(BUCKET_NAME, blob_name)  #  Get signed URL
                vm_result["download_url"] = signed_url  #  Add it to response

            #  Final response sent to frontend
            return JSONResponse(content={
                "status": "success",
                "message": vm_result.get("message"),
                "metrics_shape": vm_result.get("metrics_shape"),
                "forecast_shape": vm_result.get("forecast_shape"),
                "raw_response": vm_result,
                "download_url": vm_result.get("download_url") 
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
        print(" Outer exception:\n", traceback.format_exc()) 
        raise HTTPException(status_code=500, detail=f"Unhandled error: {str(e)}")

    # except Exception as e:
    #     print(" Outer exception:\n", traceback.format_exc())  # ### TRACEBACK LOGGING
    #     return JSONResponse(
    #         status_code=422,
    #         content={
    #             "detail": [{
    #                 "type": "value_error",
    #                 "loc": ["body", "file"],
    #                 "msg": "Expected UploadFile, got str",
    #                 "input": "string",
    #                 "ctx": {"error": {}}
    #        }]
    #         }
    #     )


# @app.get("/download/")
# def download_forecast(path: str):
#     """Streams the file from GCS back to the frontend."""
#     try:
#         bucket_name = "featurebox-ai-uploads"
#         blob_path = path.replace(f"gs://{bucket_name}/", "")
        
#         storage_client = storage.Client()
#         bucket = storage_client.bucket(bucket_name)
#         blob = bucket.blob(blob_path)
        
#         file_stream = BytesIO()
#         blob.download_to_file(file_stream)
#         file_stream.seek(0)

#         return StreamingResponse(
#             file_stream,
#             media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#             headers={"Content-Disposition": f"attachment; filename=forecast_results.xlsx"}
#         )
#     except Exception as e:
#         return {"status": "error", "message": str(e)}



    