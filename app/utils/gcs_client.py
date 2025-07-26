from google.cloud import storage
from fastapi import HTTPException
from google.auth import default 
import os

#  Get default credentials and project from Cloud Run environment
credentials, project_id = default()  # token-based credentials

# Initialize storage client with explicit credentials
storage_client = storage.Client(credentials=credentials, project=project_id)  


def upload_to_gcs(file_path: str, bucket_name: str = "featurebox-ai-uploads") -> str:
    """
    Upload a file to Google Cloud Storage
    
    Args:
        file_path: Path to the file to upload
        bucket_name: Name of the GCS bucket
    
    Returns:
        The public URL of the uploaded file
    
    Raises:
        HTTPException: If upload fails
    """
    try:
        # Uses global storage client initialized with token-based credentials
        bucket = storage_client.bucket(bucket_name)
        
        # Create a blob object from the file
        blob_name = os.path.basename(file_path)
        blob = bucket.blob(f"excel_uploads/{blob_name}")
        
        # Upload the file
        blob.upload_from_filename(file_path)
        return f"gs://{bucket_name}/{blob_name}"
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GCS Upload failed: {str(e)}")
# ─────────────────────────────────────────────────────────────
# DOWNLOAD from GCS
# ─────────────────────────────────────────────────────────────
def download_blob(bucket_name: str, source_blob_name: str, destination_file_name: str):
    """
    Download a blob from GCS to a local file.

    Args:
        bucket_name: GCS bucket name
        source_blob_name: Path to file in bucket (e.g. excel_uploads/myfile.xlsx)
        destination_file_name: Where to save locally
    """
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)

        blob.download_to_filename(destination_file_name)
        print(f" Downloaded {source_blob_name} to {destination_file_name}")
        return destination_file_name
    
    except Exception as e:
        print(f" GCS download error: {str(e)}")
        raise