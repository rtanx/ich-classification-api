from io import BytesIO

from fastapi import UploadFile
import numpy as np

from .classifier import ClassifierService
from app.schemas import requests
from app.schemas import responses


class SinogramBasedClassifierService(ClassifierService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def get_input_image(self, file: UploadFile) -> np.ndarray:
        pass
        # try:
        #     file_content = await file.read()
        #     dcm_bytes = BytesIO(file_content)
