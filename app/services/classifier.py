import os
from io import BytesIO

import anyio
from fastapi import UploadFile
import pydicom
import numpy as np

from app.internal.data_preprocessing import ImagePreparation
from app.internal.classifier import ICHModel


class Classifier():
    def __init__(self, upload_dir: str):
        self.upload_dir = upload_dir

    async def save_file(self, file: UploadFile, chunk_size: int = 4096) -> str:
        """Save an uploaded file to the specified directory."""
        fpath = os.path.join(self.upload_dir, file.filename)
        async with await anyio.open_file(fpath, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                await f.write(chunk)
        return fpath

    async def classify_from_file_upload(self, file: UploadFile) -> str:
        try:
            file_content = await file.read()
            dcm_bytes = BytesIO(file_content)
            bsb_img = ImagePreparation.read_as_array(dcm_bytes, resize=(256, 256))
        except Exception as e:
            raise ValueError(f"An error occurred: {str(e)}")

        y_preds = ICHModel(np.expand_dims(bsb_img, axis=0), training=False).numpy()
        print(y_preds.squeeze())


# labels_map = {
#     0: 'ANY', # any
#     1: 'EDH', # epidural
#     2: 'IPH', # intraparenchymal
#     3: 'IVH', # intraventricular
#     4: 'SAH', # subarachnoid
#     5: 'SDH', # subdural
# }
