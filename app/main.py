from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routers import ImageModelRouter
from app.services import ClassifierService, FilesService
from app.internal.models import ImageModel


app = FastAPI(ignore_trailing_slashes=True)

GENERATED_FILE_DIR = Path('tmp/files')
GENERATED_FILE_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=GENERATED_FILE_DIR), name="files")

# Service
classifier_service = ClassifierService(model=ImageModel)
files_service = FilesService(static_file_path=GENERATED_FILE_DIR, mounted_path="/files")

# Routers
image_model_router = ImageModelRouter(classifier_service, files_service)

app.include_router(image_model_router.routes(), prefix="/image-model", tags=['Model API'])
# app.include_router(router.router, prefix="/sinogram-model", tags=['Model API'])


@app.get("/")
async def root():
    return {"message": "Hello World"}
