from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel


class ImageModelParams(BaseModel):
    with_gradcam: Optional[bool] = False
    with_windowing: Optional[bool] = False
