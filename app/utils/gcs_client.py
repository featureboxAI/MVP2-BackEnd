from google.cloud import storage
from fastapi import HTTPException
import os

def upload_to_gcs(file_path: str, bucket_name: str) -> str:
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
        # Initialize the GCS client
        client = storage.Client()
        
        # Get the bucket
        bucket = client.bucket(bucket_name)
        
        # Create a blob object from the file
        blob_name = os.path.basename(file_path)
        blob = bucket.blob(f"excel_uploads/{blob_name}")
        
        # Upload the file
        blob.upload_from_filename(file_path)
        
        return blob.public_url

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload to GCS: {str(e)}"
        )
