from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from app.schemas.responses import ClassificationMap, ClassificationResponseUnified, ProcessedImage, Inference
from app.services import ClassifierService, FilesService
from app.schemas import requests
from app.internal.post_processing import hot_encoded_indices_to_labels


class ImageModelRouter:
    def __init__(self, classifier_service: ClassifierService, files_Service: FilesService):
        self.router = APIRouter()
        self.classifier_service = classifier_service
        self.files_service = files_Service

    async def classify(self, params: requests.ImageModelParams = Depends(), file: UploadFile = File(...)) -> ClassificationResponseUnified:
        try:
            input_img = await self.classifier_service.get_input_image(file)
            y_preds = self.classifier_service.classify(input_img)

            classification_maps = ClassificationMap.from_array(y_preds)
            resp = ClassificationResponseUnified(classification=classification_maps)
            top_label_indices = self.classifier_service.get_top_labels_indices(y_preds)

            inferences: list[Inference] = []
            if len(top_label_indices) > 0:
                top_class_labels = hot_encoded_indices_to_labels(top_label_indices)
                for class_idx, class_label in zip(top_label_indices, top_class_labels):
                    inference = Inference(label=class_label, confidence='{:.2%}'.format(y_preds[class_idx]))
                    inferences.append(inference)

                if params.with_gradcam:
                    heatmaps = self.classifier_service.compute_gradcam(input_img, input_img, top_label_indices)

                    for i, heatmap, pred_label in zip(range(0, len(inferences)), heatmaps.values(), top_class_labels):
                        fname = file.filename.split('.')[0]
                        fname = f'{fname}_{pred_label}_heatmap.png'
                        heatmap_file_path = self.files_service.write_img_array(heatmap, fname)
                        inferences[i].heatmap = heatmap_file_path

                resp.inferences = inferences

            if params.with_windowing:
                fname = file.filename.split('.')[0]
                brainw_path, subduralw_path, softw_path, stackedw_path = self.files_service.write_windowed_image(input_img, fname)
                print(f'Brain window: {brainw_path}')
                prrocessed_img = ProcessedImage(brain_window=brainw_path, subdural_window=subduralw_path, soft_window=softw_path, stacked=stackedw_path)

                resp.processed_image = prrocessed_img

        except Exception as e:
            raise HTTPException(status_code=500, detail=f'An error occurred: {str(e)}')

        return resp
        # ID_cca25a801
        # ID_9504ce37e

    def routes(self) -> APIRouter:
        self.router.post("/", response_model=ClassificationResponseUnified, response_model_exclude_none=True)(self.classify)
        return self.router
