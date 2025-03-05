from typing import List, Optional

import numpy as np

import pydicom
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from pydicom.filebase import DicomBytesIO

from crud.segmentation import (
    segment_image,  # Funções genéricas de segmentação
    segment_image2,
    inicializa_curva
)
from schemas.model_schema import SegmentedPointsResponse

router = APIRouter()


@router.post(
    "/image-segmentation",
    response_model=SegmentedPointsResponse,
    status_code=status.HTTP_200_OK,
)
async def segment_dicom(
    file: UploadFile = File(...),
    method: str = Query(
        ..., description="Método de segmentação (ex: 'segmentation' ou 'segmentation2' ou 'curva_inicial')"
    ),
    parametro1: Optional[float] = Query(
        None, description="Parâmetro 1 para segmentation2"
    ),
    parametro2: Optional[float] = Query(
        None, description="Parâmetro 2 para segmentation2"
    ),
    parametro3: Optional[float] = Query(
        None, description="Parâmetro 3 para segmentation2"
    ),
    coordenada_x: Optional[int] = Query(
        None, description="coordenada_x para o ponto central da curva_inicial"
    ),
    coordenada_y: Optional[int] = Query(
        None, description="coordenada_y para o ponto central da curva_inicial"
    ),
    raio_da_curva: Optional[int] = Query(
        None, description="raio_da_curva para curva_inicial"
    ),
    quantidade_de_pontos: Optional[int] = Query(
        None, description="quantidade_de_pontos para curva_inicial"
    ),
):
    # Verificar se o arquivo é DICOM
    if not file.filename.endswith(".dcm"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="O arquivo deve ser no formato DICOM (.dcm)",
        )

    try:
        dicom_data = await file.read()
        dicom_bytes = DicomBytesIO(dicom_data)
        pixel_array = pydicom.dcmread(dicom_bytes).pixel_array

        if method == "segmentation":
            segmented_points = segment_image(pixel_array)

        elif method == "segmentation2":
            if parametro1 is None or parametro2 is None or parametro3 is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Todos os parâmetros (parametro1, parametro2, parametro3) são obrigatórios para segmentation2",
                )
            segmented_points = segment_image2(
                pixel_array, parametro1, parametro2, parametro3
            )

        elif method == "curva_inicial":
            if coordenada_x is None or coordenada_y is None or raio_da_curva is None or quantidade_de_pontos is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Todos os parâmetros (coordenada_x, coordenada_y, raio_da_curva, quantidade_de_pontos) são obrigatórios para segmentation2",
                )
            ponto = np.array([coordenada_x, coordenada_y])
            segmented_points = inicializa_curva(
                ponto, raio_da_curva, quantidade_de_pontos
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Método de segmentação inválido. Use 'segmentation' ou 'segmentation2' ou curva_inicial.",
            )

        points = [{"x": int(x), "y": int(y)} for (x, y) in segmented_points]

        return JSONResponse(content={"points": points})

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar o arquivo DICOM: {str(e)}",
        )
