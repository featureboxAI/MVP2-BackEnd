from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from zipfile import ZipFile, BadZipFile

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "FastAPI backend is running!"}

@app.post("/upload/")
async def upload_zip(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are allowed.")

    try:
        with ZipFile(file.file) as zip_ref:
            file_list = zip_ref.namelist()  # list of all files inside the ZIP
            file_count = len(file_list)     # just count them

    except BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid or corrupt ZIP file.")

    return JSONResponse(content={
        "status": "success",
        "filename": file.filename,
        "file_count": file_count,
        "message": f"ZIP file opened successfully with {file_count} files inside"
    })
