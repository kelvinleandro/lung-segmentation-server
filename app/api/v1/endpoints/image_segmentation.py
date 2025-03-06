from typing import List, Optional

import pydicom
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from pydicom.filebase import DicomBytesIO

from crud.segmentation import (
    segment_image,  # Funções genéricas de segmentação
)

from crud.alternativas.imagem_para_base64 import imagem_para_base64
from crud.alternativas.to_hu import converte_para_hu
from crud.alternativas.hu_para_cinza import converter_hu_para_cinza

from crud.alternativas.watershed import aplicar_watershed
from crud.alternativas.lim_media_mov import aplicar_limiarizacao_media_movel
from crud.alternativas.lim_multipla import limiarizacao_multipla
from crud.alternativas.lim_prop_locais import aplicar_limiarizacao_propriedades
from crud.alternativas.sauvola import aplicar_sauvola
from crud.alternativas.div_e_fus_regioes import aplicar_divisao_e_fusao 
from crud.alternativas.otsu import aplicar_otsu



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
        ..., description="Método de segmentação (ex: 'watershed' ou 'lim_media_mov' ou 'lim_multipla' ou 'lim_prop_locais', 'sauvola', 'otsu' ou 'divisao_e_fusao')"
    ),
    aplicar_interpolacao: Optional[bool] = Query(
        None, description="Parâmetro para watershed, lim_media_mov, usar_media_global, sauvola ou lim_prop_locais"
    ),
    aplicar_morfologia: Optional[bool] = Query(
        None, description="Parâmetro  para watershed ou sauvola"
    ),
    usar_media_global: Optional[bool] = Query(
        None, description="Parâmetro  para lim_prop_locais"
    ),
    limite_var: Optional[int] = Query (
        None, description="Parâmetro para divisao_e_fusao"
    ),
    limite_media: Optional[int] = Query (
        None, description="Parâmetro para divisao_e_fusao"
    ),
    referencia_media: Optional[int] = Query (
        None, description="Parâmetro para divisao_e_fusao"
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
        ds = pydicom.dcmread(DicomBytesIO(dicom_data))  # Ler o DICOM corretamente
        pixel_array = ds.pixel_array  # Extraindo a matriz de pixels
        imagem_hu=converte_para_hu(pixel_array, ds)
        pixel_array=converter_hu_para_cinza(imagem_hu, hu_min=-1000, hu_max=2000)

        if method == "segmentation":
            segmented_points = segment_image(pixel_array)

        elif method == "watershed":
            if aplicar_interpolacao is None or aplicar_morfologia is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Todos os parâmetros (aplicar_interpolacao, aplicar_morfologia) são obrigatórios para watershed",
                )
            pulmao_contornado,contornos_validos_dict = aplicar_watershed(
                pixel_array, limiar=60, aplicar_interpolacao=aplicar_interpolacao, aplicar_morfologia=aplicar_morfologia, tamanho_kernel=3,iteracoes_morfologia=2,iteracoes_dilatacao=1,fator_dist_transform=0.3
            )

        elif method == "lim_media_mov":
            if aplicar_interpolacao is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Todos os parâmetros (aplicar_interpolacao) é obrigatório para lim_media_mov",
                )
            pulmao_contornado,contornos_validos_dict = aplicar_limiarizacao_media_movel(
                pixel_array, n=171, b=0.8, aplicar_interpolacao=aplicar_interpolacao
            )
        
        elif method == "lim_multipla":
            pulmao_contornado,contornos_validos_dict = limiarizacao_multipla(
            pixel_array, lim_hiperaeradas = (0, 8),
            lim_normalmente_aeradas = (8, 42),
            lim_pouco_aeradas = (42, 76),
            lim_nao_aeradas = (76, 93),
            lim_osso = (136, 255),
            ativacao_hiperaeradas = True,
            ativacao_normalmente_aeradas = True,
            ativacao_pouco_aeradas= True,
            ativacao_nao_aeradas = False,
            ativacao_osso = False,
            ativacao_nao_classificado = False,
        )

        elif method == "lim_prop_locais":
            if usar_media_global is None or aplicar_interpolacao is None: 
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Todos os parâmetros (usar_media_global, aplicar_interpolacao) é obrigatório para lim_prop_locais",
                )
            pulmao_contornado,contornos_validos_dict = aplicar_limiarizacao_propriedades(
                pixel_array, tamanho_janela=151, a=1.0, b=0.5, usar_media_global=usar_media_global,aplicar_interpolacao=aplicar_interpolacao
            )

        elif method == "sauvola":
            if aplicar_interpolacao is None or aplicar_morfologia is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Todos os parâmetros (aplicar_interpolacao, aplicar_morfologia) é obrigatório para aplicar_sauvola",
                )
            pulmao_contornado,contornos_validos_dict = aplicar_sauvola(
                pixel_array, tamanho_janela=171, k=0.02, aplicar_interpolacao=aplicar_interpolacao, aplicar_morfologia=aplicar_morfologia,tamanho_kernel=3,iteracoes_morfologia=1
            )
        
        elif method == "divisao_e_fusao":
            if limite_var is None or limite_media is None or referencia_media is None :
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Todos os parâmetros (limite_var, limite_media ou referencia_media ) é obrigatório para divisao_e_fusao",
                )
            pulmao_contornado,contornos_validos_dict = aplicar_divisao_e_fusao(
                pixel_array, limite_var=limite_var, limite_media=limite_media, referencia_media=referencia_media
            )  

        elif method == "otsu":
            pulmao_contornado, contornos_validos_dict = aplicar_otsu(pixel_array)
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Método de segmentação inválido. Use 'watershed' ou 'lim_media_mov' ou 'lim_multipla' ou 'lim_prop_locais', 'sauvola', 'otsu' ou 'divisao_e_fusao'.",
            )

        pulmao_contornado=imagem_para_base64(pulmao_contornado)
        return JSONResponse({"processada":pulmao_contornado, "contorno":contornos_validos_dict})

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar o arquivo DICOM: {str(e)}",
            
        )
