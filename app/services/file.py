import anyio
from fastapi import UploadFile
import os


class FileUploadHandler():
    def __init__(self, upload_dir: str):
        self.upload_dir = upload_dir

    async def save(self, file: UploadFile, chunk_size: int = 4096) -> str:
        """Save an uploaded file to the specified directory."""
        fpath = os.path.join(self.upload_dir, file.filename)
        async with await anyio.open_file(fpath, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                await f.write(chunk)
        return fpath
