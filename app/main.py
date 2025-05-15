from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "FastAPI backend is running!"}

@app.post("/upload/")
async def upload_zip(file: UploadFile = File(...)):
    # Checks for file extension
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only ZIP files are allowed.")

    # Read contents in memory
    contents = await file.read()


    return JSONResponse(content={
        "status": "success",
        "filename": file.filename,
        "message": "ZIP file received successfully"
    })
