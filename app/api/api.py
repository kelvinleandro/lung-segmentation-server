from api.v1.endpoints import image_segmentation
from fastapi import APIRouter

api_router = APIRouter()

#rota da api
api_router.include_router(image_segmentation.router, prefix="/image-segmentation", tags=["Rota para Segmentar imagem DICOM"])