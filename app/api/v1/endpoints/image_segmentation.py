from typing import List
from fastapi import APIRouter, HTTPException, status, File, UploadFile
from fastapi.responses import JSONResponse
import pydicom
from crud.segmentation import segment_image  # Função de segmentação a ser implementada
from pydicom.filebase import DicomBytesIO
from schemas.model_schema import SegmentedPointsResponse

router = APIRouter()

@router.post("/image-segmentation", response_model=SegmentedPointsResponse, status_code=status.HTTP_200_OK)
async def segment_dicom(file: UploadFile = File(...)):
    # verificar se é um arquivo DICOM
    if not file.filename.endswith('.dcm'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="O arquivo deve ser no formato DICOM (.dcm)")

    try:
        dicom_data = await file.read()
        dicom_bytes = DicomBytesIO(dicom_data)
        pixel_array = pydicom.dcmread(dicom_bytes).pixel_array

        # chama a função de segmentação (a ser implementada) crud
        segmented_points = segment_image(pixel_array)

        points = [{"x": x, "y": y} for (x, y) in segmented_points]

        return JSONResponse(content={"points": points})
    
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro ao processar o arquivo DICOM: {str(e)}")
