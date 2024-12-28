import numpy as np
import cv2
import os


class FilesService:
    def __init__(self, static_file_path: str, mounted_path: str):
        self.static_file_path = static_file_path
        self.mounted_path = mounted_path.strip('/')
        os.makedirs(self.static_file_path, exist_ok=True)

    def _generate_relative_url(self, fpaths: str | list[str]) -> str:
        if isinstance(fpaths, str):
            return os.path.join(self.mounted_path, fpaths)

        return [os.path.join(self.mounted_path, fpath) for fpath in fpaths]

    def write_img_array(self, img_array: np.ndarray, fpath: str) -> str:
        fpath = fpath if fpath.endswith('.png') else f'{fpath}.png'
        file_path = os.path.join(self.static_file_path, fpath)
        cv2.imwrite(file_path, img_array)
        return self._generate_relative_url(fpath)

    def write_windowed_image(self, input_img_array: np.ndarray, img_id: str) -> str:
        brainw_fname = f'{img_id}_brainw.png'
        subduralw_fname = f'{img_id}_subduralw.png'
        softw_fname = f'{img_id}_softw.png'
        stackedw_fname = f'{img_id}_stackedw.png'

        brainw_path = os.path.join(self.static_file_path, brainw_fname)
        subduralw_path = os.path.join(self.static_file_path, subduralw_fname)
        softw_path = os.path.join(self.static_file_path, softw_fname)
        stackedw_path = os.path.join(self.static_file_path, stackedw_fname)

        if input_img_array.dtype != np.uint8:
            input_img_array = (input_img_array * 255).astype(np.uint8)

        stacked_img = input_img_array
        [brain_img, subdural_img, soft_img] = input_img_array.transpose(2, 0, 1)

        cv2.imwrite(brainw_path, brain_img)
        cv2.imwrite(subduralw_path, subdural_img)
        cv2.imwrite(softw_path, soft_img)
        cv2.imwrite(stackedw_path, stacked_img)

        return (*self._generate_relative_url([brainw_fname, subduralw_fname, softw_fname, stackedw_fname]),)
