from typing import Optional, Literal
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from pathlib import Path
import numpy as np


class ClassificationMap(BaseModel):
    any: float
    epidural: float
    intraparenchymal: float
    intraventricular: float
    subarachnoid: float
    subdural: float

    @classmethod
    def from_array(cls, arr: np.ndarray | list):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)

        if arr.ndim != 1:
            raise ValueError('Invalid array dimension, expected 1-D array')

        if arr.shape[0] != 6:
            raise ValueError('Invalid array shape, expected 6 elements')

        return cls(any=arr[0], epidural=arr[1], intraparenchymal=arr[2], intraventricular=arr[3], subarachnoid=arr[4], subdural=arr[5])


class Inference(BaseModel):
    label: Literal['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
    confidence: str
    heatmap: Optional[HttpUrl] = None


class ProcessedImage(BaseModel):
    brain_window: Optional[Path] = None
    subdural_window: Optional[Path] = None
    soft_window: Optional[Path] = None
    stacked: Optional[Path] = None


class ClassificationResponseUnified(BaseModel):
    is_positive: bool
    classification: ClassificationMap
    inferences: Optional[list[Inference]] = None
    processed_image: Optional[ProcessedImage] = None
