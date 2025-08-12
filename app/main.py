from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
import google.auth, os, requests, tempfile, shutil, time, traceback, subprocess
import threading
from google.auth.transport.requests import Request as GoogleAuthRequest
from zipfile import ZipFile, BadZipFile
from google.cloud import storage
from fastapi.middleware.cors import CORSMiddleware
from datetime import timedelta
from fastapi import Request 
from fastapi.responses import StreamingResponse
from io import BytesIO
from google.oauth2 import service_account
from urllib.parse import quote_plus
from google.cloud import compute_v1
from datetime import datetime 
from google.cloud import storage   
from google.api_core.exceptions import GoogleAPIError
import time 

# Set Google Cloud credentials using relative path
# current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# credentials_path = os.path.join(current_dir, "service_account_key.json")

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
# print(f"[INFO] Using credentials from: {credentials_path}")

app = FastAPI()

PROJECT_ID = "fbfileuploads"                 
ZONE = "us-central1-a"                          # VM zone
INSTANCE_NAME = "instance-20250710-091502"      # VM Instance name
STATIC_VM_IP = "34.135.50.176"

def start_vm_if_needed(start_time=None):
    if start_time is None:
        start_time = time.time()
    """
    This function:
    1. Checks if the VM is running
    2. If not running → Starts the VM
    3. Waits until Uvicorn is ready by polling the /health endpoint
    """
    # 1. Creates a Compute Engine API client to interact with VM instances 
    # (handles auth + API calls so we can start/stop/get status of the VM easily)
    client = compute_v1.InstancesClient()
    
    # 2. Get the current VM status
    instance = client.get(project=PROJECT_ID, zone=ZONE, instance=INSTANCE_NAME)
    status = instance.status
    print(f"VM status: {status}")

    # 3. Start VM if not running 
    if status != "RUNNING":
        print("Starting VM...")
        operation = client.start(project=PROJECT_ID, zone=ZONE, instance=INSTANCE_NAME)
        operation.result()  # Wait for operation to complete
        print("VM start command sent") 
    else:
        print("VM already running")

    # 4. Uses static VM external IP (to check Uvicorn readiness)

    vm_ip = STATIC_VM_IP
    uvicorn_url = f"http://{vm_ip}:8000/health"  # URL for health check endpoint
    restart_url = f"http://{vm_ip}:8000/restart-uvicorn"  # restart endpoint
   
    # Initial wait after starting VM 
    initial_wait = 30  
    print(f"[{datetime.now()}] Waiting {initial_wait}s before first readiness check...")
    time.sleep(initial_wait)

    # 5. Poll until Uvicorn responds or timeout 
    for attempt in range(30):  
        try:
            response = requests.get(uvicorn_url, timeout=5)
            if response.status_code == 200:
                elapsed = time.time() - start_time
                print(f"[{datetime.now()}] Uvicorn is ready on attempt {attempt+1}!")
                return True
        except Exception as e:
            print(f"Attempt {attempt+1}: Uvicorn not ready yet... ({e})")
            print(f"[DEBUG] Exception: {repr(e)}")
        time.sleep(3)  # Wait 3 seconds before next check

    # 6. Attempt restart if still not ready
    print(f"[{datetime.now()}] Uvicorn did not respond after initial wait. Attempting restart...")
    restart_wait = 20
    print(f"[{datetime.now()}] Waiting {restart_wait}s before restart attempt...")
    time.sleep(restart_wait)

    try:
        restart_resp = requests.post(restart_url, timeout=5)
        if restart_resp.status_code == 200:
            print("[INFO] Restart triggered. Checking readiness again...")
            time.sleep(3)

            for attempt in range(15):
                try:
                    response = requests.get(uvicorn_url, timeout=5)
                    if response.status_code == 200:
                        elapsed = time.time() - start_time
                        print(f"[{datetime.now()}] Uvicorn is ready after restart (attempt {attempt+1}, elapsed: {elapsed:.2f}s)")
                        return True
                except Exception as e:
                    print(f"[{datetime.now()}] Retry {attempt+1}: Still not ready... ({e})")
                time.sleep(5)
            else:
                print(f"[{datetime.now()}] Uvicorn failed even after restart.")
                return False
        else:
            print(f"[{datetime.now()}] Restart call failed with status {restart_resp.status_code}")
            return False
    except Exception as e:
        print(f"[{datetime.now()}] Restart call failed: {e}")
        return False

@app.get("/start-vm")
def start_vm_endpoint():
    print(f"[{datetime.now()}] VM start process initiated...")
    start_time = time.time()
    ready = start_vm_if_needed(start_time)
    total_time = time.time() - start_time

    if ready:
        print(f"[{datetime.now()}] Total elapsed time: {total_time:.2f} seconds")
        return {"status": "VM and Uvicorn ready"}
    else:
        print(f"[{datetime.now()}] VM/Uvicorn failed after {total_time:.2f} seconds")
        return {"status": "error", "message": "VM started but Uvicorn failed to respond"}

# def schedule_vm_stop_after_delay(minutes=10):
#     """Schedule VM to stop after specified delay"""
#     print(f"[INFO] VM will auto-stop in {minutes} minutes")
    
#     stop_timer = threading.Timer(minutes * 60, stop_vm)
#     stop_timer.daemon = True  # Dies when main thread dies
#     stop_timer.start()

# def stop_vm():
#     """Stop the VM instance using Compute API"""
#     try:
#         print("[INFO] Auto-stopping VM now...")
#         client = compute_v1.InstancesClient()
#         operation = client.stop(project=PROJECT_ID, zone=ZONE, instance=INSTANCE_NAME)
#         operation.result()  # Wait for completion
#         print("[INFO] VM stopped successfully")
#     except Exception as e:
#         print(f"[ERROR] Failed to stop VM: {e}")

VM_STATUS_URL = "http://34.135.50.176:8000/status" 
# VM_STATUS_URL = "http://34.135.50.176:8002/status" #local testing
BUCKET_NAME = "featurebox-ai-uploads"

# Forecast status store (idle | running | completed | error)
forecast_status_store = {"status": "idle", "forecast_gcs": None, "webhook_time": None}     

# CORS CONFIGURATION
origins = [
    "https://featurebox-ai-ui-mvp-2-05.lovable.app",
    "http://localhost:3000",
    "https://featurebox-frontend-service-666676702816.us-west1.run.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=(
        r"(https:\/\/.*\.lovable\.(app|com))"   # any Lovable preview
        r"|^https:\/\/featurebox-frontend-service-666676702816\.us-west1\.run\.app$"  # Allow Cloud Run frontend
        r"|(^http:\/\/localhost(:\d+)?$)"       # localhost with optional :port
        r"|^https:\/\/[a-zA-Z0-9\-]+\.lovableproject\.com$"
        r"|(^http:\/\/localhost(:\d+)?$)"       # localhost with optional :port
        r"|(^http:\/\/127\.0\.0\.1(:\d+)?$)"    # 127.0.0.1 with optional :port
        r"|(^http:\/\/10\.0\.0\.122(:\d+)?$)"
    ),

    allow_credentials=False,   # keep False with a wildcard / regex
    allow_methods=["*"],    # or ["*"] if you’ll add GET etc.
    allow_headers=["*"],
)


# =========================================================
# Webhook endpoint from VM
# =========================================================
@app.post("/forecast-complete")
async def forecast_complete(request: Request):
    print("[DEBUG][Cloud Run] /forecast-complete endpoint triggered")
    try:
        data = await request.json()
        print(f"[DEBUG][Cloud Run] Webhook received from VM: {data}")

        # Update internal status
        forecast_status_store["status"] = data.get("status", "completed")
        forecast_status_store["forecast_gcs"] = data.get("forecast_gcs")
        forecast_status_store["webhook_time"] = time.time()

        # 10-minute timer to stop VM
        # print("[INFO] Starting 10-minute timer to auto-stop VM...")
        # schedule_vm_stop_after_delay(minutes=10)
        return {"status": "received"}
    
    except Exception as e:
        print(f"[ERROR][Cloud Run] forecast-complete error: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    
# =========================================================
# Status endpoint (Updated to verify GCS before "completed")
# =========================================================
@app.get("/status")
async def get_status():
    try:
        # 1. Check actual VM status using GCP Compute API
        compute_client = compute_v1.InstancesClient()
        instance = compute_client.get(project=PROJECT_ID, zone=ZONE, instance=INSTANCE_NAME)
        vm_status = instance.status  # e.g., "RUNNING", "TERMINATED", etc.

        # 2.Get forecast file path from store (just for GCS path)
        forecast_file_gcs = forecast_status_store.get("forecast_gcs")
        # webhook_time = forecast_status_store.get("webhook_time")

        # Check GCS only if VM is running
        if vm_status == "RUNNING" and forecast_file_gcs:
            forecast_status = "completed"

            
            #redundant
            # bucket_name = forecast_file_gcs.split("/")[2]
            # blob_path = "/".join(forecast_file_gcs.split("/")[3:])
            
            # client = storage.Client()
            # bucket = client.bucket(bucket_name)
            # blob = bucket.blob(blob_path)

            # print(f"[DEBUG] Checking GCS file existence:")
            # print(f"[DEBUG] Bucket: {bucket_name}")
            # print(f"[DEBUG] Blob path: {blob_path}")
            # print(f"[DEBUG] Full GCS URI: gs://{bucket_name}/{blob_path}")
            
            # # Check if file exists with retry logic for recent webhooks
            # file_exists = blob.exists()
            # print(f"[DEBUG] File exists: {file_exists}")
            
            # if file_exists:
            #     forecast_status = "completed"
            # else:
            #     # If webhook was recent (< 5 minutes), give more time for GCS upload
            #     current_time = time.time()
            #     if webhook_time and (current_time - webhook_time) < 300:  # 5 minutes
            #         print(f"[DEBUG] Webhook was recent ({(current_time - webhook_time):.1f}s ago), file might still be uploading")
            #         # Try a few more times with small delays
            #         for attempt in range(3):
            #             print(f"[DEBUG] Retry attempt {attempt + 1}/3 after 2s delay")
            #             time.sleep(2)
            #             if blob.exists():
            #                 print(f"[DEBUG] File found on retry attempt {attempt + 1}")
            #                 file_exists = True
            #                 break
                    
            #         if file_exists:
            #             forecast_status = "completed"
            #         else:
                #         print(f"[DEBUG] File still not found after retries, checking directory contents")
                #         # List files in the directory to see what's actually there
                #         try:
                #             blobs = list(client.bucket(bucket_name).list_blobs(prefix="excel_uploads/", max_results=10))
                #             print(f"[DEBUG] Recent files in excel_uploads/: {[b.name for b in blobs]}")
                #         except Exception as list_err:
                #             print(f"[DEBUG] Error listing directory: {list_err}")
                #         forecast_status = "running"
                # else:
                #     print(f"[DEBUG] Webhook was not recent or missing, file should exist by now")
                #     forecast_status = "running"
        
        elif vm_status == "RUNNING":
            forecast_status = "running"
        else:
            forecast_status = "idle"  # VM not running → forecast idle

        # Return VM + forecast status
        return {
            "status": forecast_status,
            "forecast_status": forecast_status,
            "vm_status": vm_status,
            "forecast_gcs": forecast_file_gcs
        }

    except GoogleAPIError as gerr:
        print(f"[ERROR] GCP API Error: {gerr}")
        return {"forecast_status": "error", "vm_status": "unknown", "message": str(gerr)}

    except Exception as e:
        print(f"[ERROR] Failed to get status: {e}")
        return {"forecast_status": "error", "vm_status": "unknown", "message": str(e)}

   
# =========================================================
# Download forecast
# =========================================================      
@app.get("/download-forecast")
async def download_forecast():
    try:
         # 1. Get the GCS path from forecast_status_store
        forecast_file_gcs = forecast_status_store.get("forecast_gcs")
        if not forecast_file_gcs:
            raise HTTPException(status_code=404, detail="No forecast file available")

        # 2. Extract bucket & file path from GCS URI
        bucket_name = forecast_file_gcs.split("/")[2]
        blob_path = "/".join(forecast_file_gcs.split("/")[3:])

        # 3. Connect to GCS
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # 4. Wait until file exists in GCS before downloading (VM code has retry to check for file's presence in GCS)
        # max_retries = 5  
        # retry_delay = 2  
        # for attempt in range(max_retries): 
        #     if blob.exists(): 
        #         print(f"[DEBUG] File found in GCS after {attempt+1} checks")  
        #         break  
        #     else:  
        #         print(f"[DEBUG] File not found yet in GCS. Retrying {attempt+1}/{max_retries}")  
        #         time.sleep(retry_delay)  
        # else:  
        #     raise HTTPException(status_code=404, detail="Forecast file not yet available in GCS")
          
         # 5. Stream directly from GCS (no /tmp download)
        print(f"[DEBUG] Streaming forecast file directly from GCS: {forecast_file_gcs}")
        data = blob.download_as_bytes()
        filename = os.path.basename(blob_path)
        
        headers = {
            "Content-Disposition": f"attachment; filename=\"{filename}\""
        }

        # Reset status so next forecast starts clean
        # forecast_status_store["status"] = "idle"
        # forecast_status_store["forecast_gcs"] = None
        # print("[DEBUG] Status reset to idle after download")

        # 6. Stream directly to user's browser
        return StreamingResponse(
            BytesIO(data),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers
        )
    
    except Exception as e:
        print(f"[ERROR] Failed to download forecast: {e}")
        return {"status": "error", "message": str(e)}


# ------------------------------------------------------------------
# Upload helper – uploads file to GCS
# ------------------------------------------------------------------
def upload_to_gcs(file_path: str, bucket_name=BUCKET_NAME) -> str:
    """
    Uploads a local file (from Cloud Run /tmp) to a Google Cloud Storage bucket.
    
    Args:
        file_path (str): Local path of the file in Cloud Run (e.g., /tmp/core.xlsx)
        bucket_name (str): GCS bucket name (default: BUCKET_NAME constant)
    
    Returns:
        str: Relative path of uploaded file inside the bucket (e.g., excel_uploads/core.xlsx)
    """
    try:
        client = storage.Client()  # Create a GCS client using Cloud Run's service account credentials
        print(f"[INFO] Cloud Run project: {client.project}") 
        try:
            sa_email = client._credentials.service_account_email  
            print(f"[INFO] Using service account: {sa_email}")    
        except AttributeError:
            print("[WARNING] Could not determine service account email") 
    
        bucket = client.bucket(bucket_name)
        
        blob_name = f"HP_uploads/{os.path.basename(file_path)}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path) #Upload the local file from Cloud Run's /tmp to the bucket location

        print(f"[INFO] Uploaded {blob_name} to GCS")
        return blob_name
    except Exception as e:
        # Explicitly log Forbidden for missing IAM permissions
        if "403" in str(e) or "Forbidden" in str(e):
            print(f"[ERROR] Permission denied for service account: {str(e)}")
        else:
            print(f"[ERROR] GCS Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"GCS Upload failed: {str(e)}")


# ------------------------------------------------------------------
# VM trigger – posts a JSON payload to forecast‑VM and returns
#              its JSON response (or raises HTTPException on error)
# ------------------------------------------------------------------

def trigger_forecast_on_vm(core_gcs, cons_gcs=None, az_gcs=None):
    if not core_gcs:
        raise ValueError("core_gcs must be provided to trigger_forecast_on_vm")

    vm_url = os.getenv("FORECAST_VM_ENDPOINT", "http://34.135.50.176:8000/run-forecast")

    #local testing
    # vm_url = os.getenv("FORECAST_VM_ENDPOINT", "http://34.135.50.176:8002/run-forecast")
    
    print(f"[DEBUG] VM URL: {vm_url}")
    print(f"[DEBUG] Making POST request to VM endpoint")

    payload = {
        "core_path": core_gcs,
        "cons_path": cons_gcs,
        "az_path": az_gcs
    }
    clean_payload = {k: v for k, v in payload.items() if v}
    
    print("[DEBUG] Payload being sent to VM:", clean_payload)
    print("[DEBUG] Request method: POST")
    print("[DEBUG] Request headers: {'Content-Type': 'application/json'}")

    try:
        # Explicitly set headers to ensure proper POST request
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        response = requests.post(
            vm_url, 
            json=clean_payload, 
            headers=headers,
            timeout=60
        )
        
        print(f"[DEBUG] VM response status: {response.status_code}")
        print(f"[DEBUG] VM response headers: {dict(response.headers)}")
        print(f"[DEBUG] VM response text: {response.text}")

        # Handle different response status codes
        if response.status_code == 405:
            print("[ERROR] VM returned 405 Method Not Allowed - endpoint might expect different HTTP method")
            return {"status": "error", "message": "VM endpoint method not allowed", "status_code": 405}
        
        if response.status_code >= 400:
            print(f"[ERROR] VM returned error status: {response.status_code}")
            return {"status": "error", "message": f"VM error: {response.status_code}", "raw": response.text}

        # Try to parse JSON response
        try:
            return response.json()
        except Exception as json_err:
            print(f"[ERROR] Could not parse JSON from VM: {json_err}")
            return {"status": "unknown", "raw": response.text}

    except requests.exceptions.ConnectionError as e:
        print(f"[ERROR] Connection failed to VM: {e}")
        raise HTTPException(status_code=500, detail=f"VM connection failed: {str(e)}")
    except requests.exceptions.Timeout as e:
        print(f"[ERROR] Request timeout to VM: {e}")
        raise HTTPException(status_code=500, detail=f"VM request timeout: {str(e)}")
    except Exception as e:
        print(f"[ERROR] VM call failed: {e}")
        print(f"[ERROR] Exception type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"VM request failed: {str(e)}")

# ------------------------------------------------------------------
# FastAPI route – handles the ZIP upload
# ------------------------------------------------------------------

@app.post("/upload/")
async def upload_zip(file: UploadFile = File(...)):
    print("[INFO] ===== Starting file upload process =====")
    print(f"[INFO] Received file: {file.filename}")
    print(f"[INFO] Content type: {file.content_type}")
    print(f"[INFO] File size: {file.size if hasattr(file, 'size') else 'Unknown'}")

    
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are allowed.")

    print(f"[INFO] Creating temporary directory for file processing")
    tmpdir = tempfile.mkdtemp(prefix="zip_extract_")
    zip_path = os.path.join(tmpdir, file.filename)
    print(f"[INFO] Temporary directory: {tmpdir}")
    print(f"[INFO] ZIP file path: {zip_path}")

    forecast_status_store["status"] = "running"   # Update status
    forecast_status_store["forecast_gcs"] = None
    # forecast_status_store["webhook_time"] = None  # Reset webhook time

    try:
            print(f"[INFO] Saving uploaded file to temporary location")
            file.file.seek(0)
            with open(zip_path, "wb") as buffer:
                while chunk := file.file.read(1024 * 1024):
                    buffer.write(chunk)
            print(f"[INFO] File saved successfully to {zip_path}")

            print(f"[INFO] Extracting ZIP file contents")
            with ZipFile(zip_path) as z:
                if z.testzip() is not None:
                    raise HTTPException(status_code=400, detail="Corrupt ZIP file detected.")
                z.extractall(tmpdir)
            print(f"[INFO] ZIP file extracted successfully")

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
                uploaded_files.append(os.path.basename(excel_file))
                
                if "amazon" in fname or "az" in fname:
                    az_file = excel_file
                elif any(kw in fname for kw in ["cons", "spins", "demand"]):
                    cons_file = excel_file
                elif forecast_file is None or any(kw in fname for kw in ["core", "allsku", "sprout"]):
                    forecast_file = excel_file

            if not forecast_file:
                raise HTTPException(status_code=400, detail="Could not identify forecast file (core/allsku/sprout) from uploaded ZIP")
            
             # DEBUG: Print what we're sending to VM
            print(" [BACKEND] Triggering forecast on VM with:")
            print(" - Forecast file:", forecast_file)
            print(" - Cons file:", cons_file)
            print(" - Amazon file:", az_file)
            print(" [DEBUG] Selected forecast_file:", forecast_file)

            # Ensure VM + Uvicorn ready
            ready = start_vm_if_needed()   # Check readiness before forecast
            if not ready:
                return {"status": "error", "message": "VM/Uvicorn not ready"}  

            # Upload files to GCS (only once)
            gcs_core = upload_to_gcs(forecast_file, BUCKET_NAME)
            print(" [DEBUG] Uploaded forecast file to GCS as:", gcs_core)
            gcs_cons = upload_to_gcs(cons_file, BUCKET_NAME) if cons_file else None
            gcs_az   = upload_to_gcs(az_file, BUCKET_NAME) if az_file else None

            print("[DEBUG] About to call trigger_forecast_on_vm...")
            # Call VM with JSON body of GCS paths
            vm_result = trigger_forecast_on_vm(gcs_core, gcs_cons, gcs_az)
            print("[DEBUG] Result from VM:", vm_result)

            # Log response
            print("[BACKEND] VM responded with:", vm_result)

            #  Generate signed download URL from forecast_gcs if returned
            if vm_result.get("status") == "success" and "forecast_gcs" in vm_result:
                gcs_uri   = vm_result["forecast_gcs"]
                blob_path = gcs_uri.replace(f"gs://{BUCKET_NAME}/", "")
                
                # Use default credentials in Cloud Run (no JSON file path)
                client = storage.Client()  
                print(f"[INFO] Cloud Run project: {client.project}")  
                try:
                    sa_email = client._credentials.service_account_email 
                    print(f"[INFO] Using service account: {sa_email}")   
                except AttributeError:
                    print("[WARNING] Could not determine service account email")  
               
                blob   = client.bucket(BUCKET_NAME).blob(blob_path)  
                
                data      = blob.download_as_bytes()                               
                filename  = os.path.basename(blob_path)                        
                headers   = {                                                     
                    "Content-Disposition": f"attachment; filename=\"{filename}\""  
                }                                                                      
                return StreamingResponse(                                          
                    BytesIO(data),                                               
                    media_type=(                                                   
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    ),                                                             
                    headers=headers,                                              
                )                                                                

            return JSONResponse({"status": "started", "message": "Forecast started successfully"})  
           

    except BadZipFile:
        raise HTTPException(400, "Invalid ZIP file.")
        
    except Exception as e:
        error_details = traceback.format_exc()  # Capture full traceback
        print("[ERROR] Full traceback:\n", error_details)  # This shows in Cloud Run logs
        raise HTTPException(
            status_code=500,
            detail=f"Backend error: {str(e)}"  # Show real error in response
        )
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
            print(f"[INFO] Cleaned up temporary directory: {tmpdir}")
        except Exception as cleanup_error:
            print(f"[WARNING] Failed to cleanup temp dir: {cleanup_error}")

# DEBUG ENDPOINTS
@app.get("/")
async def root():
    return {"message": "FastAPI backend is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {
        "status": "healthy",
        "timestamp": "2025-07-28T19:31:06Z",
        "service": "featurebox-ai-backend"
    }

# @app.get("/whoami") 
# async def whoami():
#     """
#     Returns Cloud Run's service account and project ID.
#     Helps verify correct credentials before GCS calls.
#     """
#     try:
#         client = storage.Client()
#         project_id = client.project
#         try:
#             sa_email = client._credentials.service_account_email
#         except AttributeError:
#             sa_email = "Unable to detect (possibly default credentials)"
        
#         return {
#             "project_id": project_id,
#             "service_account": sa_email,
#             "status": "ok"
#         }
#     except Exception as e:
#         return {
#             "status": "error",
#             "error": str(e)
#         }

# @app.get("/test-gcs")
# async def test_gcs():
#     """
#     Test GCS connectivity and permissions
#     """
#     try:
#         client = storage.Client()
#         bucket = client.bucket(BUCKET_NAME)
        
#         # Test if we can list blobs (read permission)
#         blobs = list(bucket.list_blobs(max_results=1))
        
#         return {
#             "status": "success",
#             "bucket": BUCKET_NAME,
#             "can_list_blobs": True,
#             "blob_count": len(blobs)
#         }
#     except Exception as e:
#         return {
#             "status": "error",
#             "error": str(e),
#             "bucket": BUCKET_NAME
#         }

# @app.get("/test-vm")
# async def test_vm():
#     """
#     Test VM connectivity and endpoint
#     """
#     vm_url = os.getenv("FORECAST_VM_ENDPOINT", "http://34.135.50.176:8000/run-forecast")
    

#     # Test with a simple payload
#     test_payload = {
#         "core_path": "test/path.xlsx",
#         "cons_path": None,
#         "az_path": None
#     }
    
#     try:
#         headers = {'Content-Type': 'application/json'}
#         response = requests.post(vm_url, json=test_payload, headers=headers, timeout=10)
#         return {
#             "status": "success",
#             "vm_status_code": response.status_code,
#             "vm_response": response.text,
#             "vm_url": vm_url,
#             "request_method": "POST",
#             "request_headers": headers
#         }
#     except Exception as e:
#         return {
#             "status": "error",
#             "error": str(e),
#             "vm_url": vm_url,
#             "request_method": "POST"
#         }



