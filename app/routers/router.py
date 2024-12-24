import os

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.file import FileUploadHandler
from app.services.classifier import Classifier

router = APIRouter()
UPLOAD_DIR = os.path.join(os.getcwd(), "tmp")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# file_handler = FileUploadHandler(UPLOAD_DIR)
classifier = Classifier(UPLOAD_DIR)

@router.post("/classify")
async def classify(file: UploadFile = File(...)):
    try:
        await classifier.classify_from_file_upload(file)
        # fpath = await file_handler.save(file)
        # return {
        #     "message": "File uploaded successfully",
        #     "file_path": fpath
        # }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f'An error occurred: {str(e)}'
        )
