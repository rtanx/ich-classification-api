from io import BytesIO

from fastapi import UploadFile
import numpy as np

from app.internal.data_preprocessing import ImagePreparation
from app.internal.post_processing import GradCAM


class ClassifierService():

    def __init__(self, model):
        self.model = model
        self.gradcam = GradCAM(self.model, 'top_conv')

    async def get_input_image(self, file: UploadFile) -> np.ndarray:
        try:
            file_content = await file.read()
            dcm_bytes = BytesIO(file_content)
            bsb_img = ImagePreparation.read_as_array(dcm_bytes, resize=(256, 256))
        except Exception as e:
            raise ValueError(f"An error occurred: {str(e)}")

        return bsb_img

    def classify(self, img: np.ndarray) -> np.ndarray:
        if img.ndim != 4:
            img = np.expand_dims(img, axis=0)

        y_preds = self.model.predict(img, verbose=0)
        return y_preds.squeeze()

    def compute_gradcam(self, img: np.ndarray, original_img: np.ndarray, indices: list | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        heatmaps = self.gradcam.generate_heatmaps(img, original_img, indices)
        return heatmaps

    def get_top_labels_indices(self, y_preds: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return np.argwhere(y_preds > threshold).squeeze()


# labels_map = {
#     0: 'ANY', # any
#     1: 'EDH', # epidural
#     2: 'IPH', # intraparenchymal
#     3: 'IVH', # intraventricular
#     4: 'SAH', # subarachnoid
#     5: 'SDH', # subdural
# }
