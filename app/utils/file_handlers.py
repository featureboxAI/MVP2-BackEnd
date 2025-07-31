import os
import tempfile
import zipfile
from fastapi import UploadFile, HTTPException
from app.utils.gcs_client import upload_to_gcs    #importing a func from a different file
from google.cloud import storage

async def process_zip_file(file: UploadFile, bucket_name: str = None):
    """
    Process a zip file containing Excel files and upload them to GCS.
    
    Args:
        file: The uploaded ZIP file
        bucket_name: The GCS bucket name where files should be uploaded
    """
    if not bucket_name:
        raise HTTPException(
            status_code=400,
            detail="GCS bucket name is required"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "uploaded.zip")

        # Save uploaded zip to temp dir
        with open(zip_path, "wb") as f:
            f.write(await file.read())

        # Validate zip file
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Check for corruption
                if zip_ref.testzip() is not None:
                    raise HTTPException(status_code=400, detail="Corrupt ZIP file detected.")
                
                # Extract files
                zip_ref.extractall(temp_dir)
                file_list = zip_ref.namelist()

        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file format.")

        uploaded_files = []
        excel_files = [f for f in file_list if f.lower().endswith('.xlsx') and not f.startswith('__MACOSX')]

        if not excel_files:
            raise HTTPException(status_code=400, detail="No Excel (.xlsx) files found in the ZIP file.")

        for fname in excel_files:
            if fname.endswith(".xlsx"):
                excel_path = os.path.join(temp_dir, fname)
                
                try:
                    # Upload Excel file to GCS
                    client = storage.Client()
                    bucket = client.bucket(bucket_name)
                    upload_to_gcs(excel_path, bucket_name)
                    uploaded_files.append(fname)
                except Exception as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Failed to upload {fname} to GCS: {str(e)}"
                    )

        return {
            "status": "success",
            "message": f"{len(uploaded_files)} Excel files uploaded to GCS.",
            "uploaded_files": uploaded_files
        }
