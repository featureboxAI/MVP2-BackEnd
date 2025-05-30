from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from zipfile import ZipFile, BadZipFile
import os
import tempfile
from google.cloud import storage
import pandas as pd
from io import BytesIO

# Set Google Cloud credentials using relative path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
credentials_path = os.path.join(current_dir, "service_account_key.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

app = FastAPI()

# Create directory for temporary CSV files
CSV_OUTPUT_DIR = "converted_csvs"
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "FastAPI backend is running!"}

@app.post("/upload/")
async def upload_zip(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are allowed.")

    # Create temporary directory for zip extraction
    tmpdir = tempfile.mkdtemp(prefix="zip_extract_")
    zip_path = os.path.join(tmpdir, file.filename)

    try:
        # Save uploaded ZIP file to disk
        file.file.seek(0)
        with open(zip_path, "wb") as buffer:
            while chunk := file.file.read(1024 * 1024):
                buffer.write(chunk)

        # Extract and validate ZIP
        with ZipFile(zip_path) as z:
            if z.testzip() is not None:
                raise HTTPException(status_code=400, detail="Corrupt ZIP file detected.")
            
            # Get list of Excel files (excluding macOS hidden files)
            file_list = [f for f in z.namelist() if not (f.startswith("__MACOSX/") or f.endswith(".DS_Store"))]
            excel_files = [f for f in file_list if f.lower().endswith((".xlsx", ".xls"))]

            if not excel_files:
                raise HTTPException(status_code=400, detail="No Excel files found in ZIP.")

            saved_files = []
            gcs_uris = []

            # Process each Excel file
            for excel_file in excel_files:
                with z.open(excel_file) as excel_fp:
                    # Read Excel and convert to CSV
                    df = pd.read_excel(BytesIO(excel_fp.read()))
                    base_name = os.path.splitext(os.path.basename(excel_file))[0]
                    csv_path = os.path.join(CSV_OUTPUT_DIR, f"{base_name}.csv")
                    df.to_csv(csv_path, index=False)
                    saved_files.append(csv_path)

                    try:
                        # Debug: Print credentials info
                        client = storage.Client()
                        print(f"Using project: {client.project}")
                        print(f"Authenticated as: {client._credentials.service_account_email}")
                        
                        bucket = client.bucket("featurebox-ai-uploads")
                        print(f"Trying to upload to bucket: {bucket.name}")
                        
                        # Upload to GCS in 'converted_csvs' folder
                        gcs_path = f"converted_csvs/{base_name}.csv"
                        blob = bucket.blob(gcs_path)
                        blob.upload_from_filename(csv_path)
                        gcs_uri = f"gs://featurebox-ai-uploads/{gcs_path}"
                        gcs_uris.append(gcs_uri)
                        print(f"Successfully uploaded {gcs_path} to GCS bucket")
                    except Exception as e:
                        print(f"Upload error details: {str(e)}")
                        raise HTTPException(status_code=500, detail=f"GCS Upload failed: {str(e)}")

            return JSONResponse(content={
                "status": "success",
                "filename": file.filename,
                "total_files_in_zip": len(file_list),
                "excel_files_converted": len(saved_files),
                "csv_files_saved": saved_files,
                "csv_files_gcs": gcs_uris,
                "message": f"Converted {len(saved_files)} Excel files to CSV and uploaded to GCS."
            })

    except BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid or corrupt ZIP file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        # Clean up
        if os.path.exists(tmpdir):
            try:
                import shutil
                shutil.rmtree(tmpdir)
            except Exception:
                pass
