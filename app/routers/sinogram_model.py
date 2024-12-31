from fastapi import APIRouter


class SinogramModelRouter():
    def __init__(self, classifier_service, file_service):
        self.router = APIRouter()
        self.classifier_service = classifier_service
        self.file_service = file_service
