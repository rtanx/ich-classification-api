from typing import Tuple, Literal, TypeAlias
import os
from io import BytesIO

import pydicom
import cv2
import numpy as np
from pydicom.fileutil import PathType
from pydicom.filebase import ReadableBuffer


WindowType: TypeAlias = Literal['brain', 'subdural', 'soft']


class ImagePreparation():
    @staticmethod
    def correct_dcm(dcm: pydicom.FileDataset):
        x = dcm.pixel_array
        x = x + 1000
        px_mode = 4096
        x[x >= px_mode] = x[x >= px_mode] - px_mode
        dcm.PixelData = x.tobytes()
        dcm.RescaleIntercept = -1000

    @staticmethod
    def window_image(dcm: pydicom.FileDataset, window_center: int, window_width: int):
        if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
            ImagePreparation.correct_dcm(dcm)

        # Pixel to Hounsfield Unit (HU)
        # HU=(Pixel Value×RescaleSlope)+RescaleIntercept
        img = dcm.pixel_array
        img = img * dcm.RescaleSlope + dcm.RescaleIntercept
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img = np.clip(img, img_min, img_max)

        return img

    @staticmethod
    def get_windowed_image(dcm: pydicom.FileDataset, window: WindowType = 'brain') -> np.ndarray:
        im = None
        match window:
            case 'brain':
                brain_img = ImagePreparation.window_image(dcm, 40, 80)
                brain_img = (brain_img - 0) / 80
                im = brain_img
            case 'subdural':
                subdural_img = ImagePreparation.window_image(dcm, 80, 200)
                subdural_img = (subdural_img - (-20)) / 200
                im = subdural_img
            case 'soft':
                soft_img = ImagePreparation.window_image(dcm, 40, 380)
                soft_img = (soft_img - (-150)) / 380
                im = soft_img
            case _:
                raise ValueError('invalid window argument')

        return im

    @staticmethod
    def bsb_window(dcm):
        brain_img = ImagePreparation.get_windowed_image(dcm, window='brain')
        subdural_img = ImagePreparation.get_windowed_image(dcm, window='subdural')
        soft_img = ImagePreparation.get_windowed_image(dcm, window='soft')

        bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

        return bsb_img

    @staticmethod
    def read_as_array(fp: PathType | BytesIO | ReadableBuffer, resize: Tuple[int, int] = (256, 256)) -> np.ndarray:
        img = None
        try:
            dcm = pydicom.dcmread(fp)
            img = ImagePreparation.bsb_window(dcm)
        except Exception as e:
            raise ValueError(
                f"An error occurred while processing dicom: {str(e)}")

        if resize is not None:
            img = cv2.resize(img, resize[:2], interpolation=cv2.INTER_LINEAR)

        return img.astype(np.float32)

    @staticmethod
    def read_as_sinogram_array(fp: PathType | BytesIO | ReadableBuffer, resize: Tuple[int, int] = (256, 256)):
        pass
