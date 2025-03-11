from typing import Any, Dict
import json
import traceback

import pydicom
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status, Form
from fastapi.responses import JSONResponse
from pydicom.filebase import DicomBytesIO

from crud.segmentation import MCACrisp # Classe do método de segmentação principal

from crud.alternativas.imagem_para_base64 import imagem_para_base64
from crud.alternativas.to_hu import converte_para_hu
from crud.alternativas.hu_para_cinza import converter_hu_para_cinza
from crud.alternativas.converte_str_json import converte_param_preprocess, converter_parametros_para_tipos
from crud.alternativas.remove_fundo import remove_fundo

from crud.alternativas.watershed import aplicar_watershed
from crud.alternativas.lim_media_mov import aplicar_limiarizacao_media_movel
from crud.alternativas.lim_multipla import limiarizacao_multipla
from crud.alternativas.lim_prop_locais import aplicar_limiarizacao_propriedades
from crud.alternativas.sauvola import aplicar_sauvola
from crud.alternativas.div_e_fus_regioes import aplicar_divisao_e_fusao 
from crud.alternativas.otsu import aplicar_otsu
from crud.alternativas.aplicar_filtros import aplicar_filtros
from crud.alternativas.crescimento_regioes_fora import crescimento_regioes_fora


router = APIRouter()


@router.post(
    "/image-segmentation",
    response_model=Dict[str, Any],
    status_code=status.HTTP_200_OK,
)
async def segment_dicom(
    file: UploadFile = File(...),
    method: str = Query(..., description="Método de segmentação"),
    params: str = Form(...),
):
    # Desserializando a string JSON para dicionário
    
    params_dict = json.loads(params)
    preprocessing_params = params_dict.get("preprocessing_params", {})
    segmentation_params = params_dict.get("segmentation_params", {})
    postprocessing_params = params_dict.get("postprocessing_params", {})

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

        if preprocessing_params:
            missing_params = [key for key, value in preprocessing_params.items() if value is None]

            if missing_params:
                raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Os seguintes parâmetros estão ausentes ou nulos: {', '.join(missing_params)}",)

            preprocessing_params=converte_param_preprocess(preprocessing_params)
            pixel_array=aplicar_filtros(pixel_array, 
                preprocessing_params['aplicar_desfoque_media'],preprocessing_params['aplicar_desfoque_gaussiano'],
                preprocessing_params['aplicar_desfoque_mediana'],preprocessing_params['tamanho_kernel'],
                preprocessing_params['sigma'])

        if method == "segmentation":
            missing_params = [key for key, value in preprocessing_params.items() if value is None]

            if missing_params:
                raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Os seguintes parâmetros estão ausentes ou nulos: {', '.join(missing_params)}",)
            
            mca_esquerdo  = MCACrisp(
                imagem_hu=imagem_hu,y_min=180,y_max=360,
                x_min=256,x_max=512,quantidade_pixels = segmentation_params['quantidade_pixels'],
                raio=segmentation_params['raio'],w_cont=segmentation_params['w_cont'],
                w_adapt=segmentation_params['w_adapt'],d_max=segmentation_params['d_max'],
                area_de_busca=segmentation_params['area_de_busca'],alpha=segmentation_params['alpha'],
                early_stop=segmentation_params['early_stop'])
            
            #segmentando o primeiro pulmao
            for curva in mca_esquerdo.process(max_iterations=segmentation_params['max_iterations']):
                pass    
            contorno_0 = mca_esquerdo.curvas[-1]
            
            #crisp para o pulmão direito
            mca_direito = MCACrisp(
                imagem_hu=imagem_hu, y_min=180, y_max=360,
                x_min=0, x_max=256, quantidade_pixels=segmentation_params['quantidade_pixels'],
                raio=segmentation_params['raio'], w_cont=segmentation_params['w_cont'],
                w_adapt=segmentation_params['w_adapt'], d_max=segmentation_params['d_max'],
                area_de_busca=segmentation_params['area_de_busca'], alpha=segmentation_params['alpha'],
                early_stop=segmentation_params['early_stop']
            )

            #pulmão direito
            for curva in mca_direito.process(max_iterations=segmentation_params['max_iterations']):
                pass    
            contorno_1 = mca_direito.curvas[-1]

            contornos_validos = {"contorno_0": contorno_0, "contorno_1": contorno_1}

            for key in contornos_validos:
                contornos_validos[key] = contornos_validos[key].tolist()
            todos_os_contornos = {}

        elif method == "watershed":
            if segmentation_params:
                missing_params = [key for key, value in segmentation_params.items() if value is None]

                if missing_params:
                    raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Os seguintes parâmetros estão ausentes ou nulos: {', '.join(missing_params)}",)
            
                segmentation_params=converter_parametros_para_tipos(segmentation_params)
                print(f"limiar={segmentation_params['limiar']}")
                print(f"aplicar_interpolacao={segmentation_params['aplicar_interpolacao']}")
                print(f"aplicar_morfologia={segmentation_params['aplicar_morfologia']}")
                print(f"tamanho_kernel={segmentation_params['tamanho_kernel']}")
                print(f"iteracoes_morfologia={segmentation_params['iteracoes_morfologia']}")
                print(f"iteracoes_dilatacao={segmentation_params['iteracoes_dilatacao']}")
                print(f"fator_dist_transform={segmentation_params['fator_dist_transform']}")
            
             
                mascara_segmentada= aplicar_watershed(pixel_array,
                    segmentation_params['limiar'],segmentation_params['aplicar_interpolacao'],
                    segmentation_params['aplicar_morfologia'], segmentation_params['tamanho_kernel'],
                    segmentation_params['iteracoes_morfologia'],segmentation_params['iteracoes_dilatacao'],
                    segmentation_params['fator_dist_transform'])
                todos_os_contornos,contornos_validos=remove_fundo(mascara_segmentada,postprocessing_params['area_minima'])
        
        elif method == "lim_media_mov":
            if segmentation_params:
                missing_params = [key for key, value in segmentation_params.items() if value is None]
            
                if missing_params:
                    raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Os seguintes parâmetros estão ausentes ou nulos: {', '.join(missing_params)}",)
                segmentation_params=converter_parametros_para_tipos(segmentation_params)
            
                print(f"n={segmentation_params['n']}")
                print(f"b={segmentation_params['b']}")
                print(f"aplicar_interpolacao={segmentation_params['aplicar_interpolacao']}")

                mascara_segmentada = aplicar_limiarizacao_media_movel(
                    pixel_array, segmentation_params['n'],segmentation_params['b'],segmentation_params['aplicar_interpolacao'])
                todos_os_contornos,contornos_validos=remove_fundo(mascara_segmentada,postprocessing_params['area_minima'])
        
        elif method == "lim_multipla":
            if segmentation_params:
                missing_params = [key for key, value in segmentation_params.items() if value is None]

                if missing_params:
                    raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Os seguintes parâmetros estão ausentes ou nulos: {', '.join(missing_params)}",)
                segmentation_params=converter_parametros_para_tipos(segmentation_params)
                print(f"lim_hiperaeradas={segmentation_params['lim_hiperaeradas']}")
                print(f"lim_normalmente_aeradas={segmentation_params['lim_normalmente_aeradas']}")
                print(f"lim_pouco_aeradas={segmentation_params['lim_pouco_aeradas']}")
                print(f"lim_nao_aeradas={segmentation_params['lim_nao_aeradas']}")
                print(f"lim_osso={segmentation_params['lim_osso']}")
                print(f"ativacao_hiperaeradas={segmentation_params['ativacao_hiperaeradas']}")
                print(f"ativacao_normalmente_aeradas={segmentation_params['ativacao_normalmente_aeradas']}")
                print(f"ativacao_pouco_aeradas={segmentation_params['ativacao_pouco_aeradas']}")
                print(f"ativacao_nao_aeradas={segmentation_params['ativacao_nao_aeradas']}")
                print(f"ativacao_osso={segmentation_params['ativacao_osso']}")
                print(f"ativacao_nao_classificado={segmentation_params['ativacao_nao_classificado']}")
                mascara_segmentada = limiarizacao_multipla(
                    pixel_array, segmentation_params['lim_hiperaeradas'],
                    segmentation_params['lim_normalmente_aeradas'],segmentation_params['lim_pouco_aeradas'],
                    segmentation_params['lim_nao_aeradas'],segmentation_params['lim_osso'],
                    segmentation_params['ativacao_hiperaeradas'],segmentation_params['ativacao_normalmente_aeradas'],
                    segmentation_params['ativacao_pouco_aeradas'],segmentation_params['ativacao_nao_aeradas'],
                    segmentation_params['ativacao_osso'],segmentation_params['ativacao_nao_classificado'])
                todos_os_contornos,contornos_validos=remove_fundo(mascara_segmentada,postprocessing_params['area_minima'])
        
        elif method == "lim_prop_locais":
            if segmentation_params:
                missing_params = [key for key, value in segmentation_params.items() if value is None]

                if missing_params:
                    raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Os seguintes parâmetros estão ausentes ou nulos: {', '.join(missing_params)}",)
                segmentation_params=converter_parametros_para_tipos(segmentation_params)
                print(f"tamanho_janela={segmentation_params['tamanho_janela']}")
                print(f"a={segmentation_params['a']}")
                print(f"b={segmentation_params['b']}")
                print(f"usar_media_global={segmentation_params['usar_media_global']}")
                print(f"aplicar_interpolacao={segmentation_params['aplicar_interpolacao']}")
                mascara_segmentada = aplicar_limiarizacao_propriedades(
                    pixel_array, segmentation_params['tamanho_janela'],
                    segmentation_params['a'],segmentation_params['b'],
                    segmentation_params['usar_media_global'],segmentation_params['aplicar_interpolacao'])
                todos_os_contornos,contornos_validos=remove_fundo(mascara_segmentada,postprocessing_params['area_minima'])
        
        elif method == "sauvola":
           if segmentation_params:
                missing_params = [key for key, value in segmentation_params.items() if value is None]

                if missing_params:
                    raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Os seguintes parâmetros estão ausentes ou nulos: {', '.join(missing_params)}",)
                segmentation_params=converter_parametros_para_tipos(segmentation_params)
                print(f"tamanho_janela={segmentation_params['tamanho_janela']}")
                print(f"k={segmentation_params['k']}")
                print(f"aplicar_interpolacao={segmentation_params['aplicar_interpolacao']}")
                print(f"aplicar_morfologia={segmentation_params['aplicar_morfologia']}")
                print(f"tamanho_kernel={segmentation_params['tamanho_kernel']}")
                print(f"iteracoes_morfologia={segmentation_params['iteracoes_morfologia']}")
                mascara_segmentada = aplicar_sauvola(
                    pixel_array,segmentation_params['tamanho_janela'],
                    segmentation_params['k'],segmentation_params['aplicar_interpolacao'],
                    segmentation_params['aplicar_morfologia'],segmentation_params['tamanho_kernel'],
                    segmentation_params['iteracoes_morfologia'])
                todos_os_contornos,contornos_validos=remove_fundo(mascara_segmentada,postprocessing_params['area_minima'])
        
        elif method == "divisao_e_fusao":
            if segmentation_params:
                missing_params = [key for key, value in segmentation_params.items() if value is None]

                if missing_params:
                    raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Os seguintes parâmetros estão ausentes ou nulos: {', '.join(missing_params)}",)
                segmentation_params=converter_parametros_para_tipos(segmentation_params)
                print(f"limite_var={segmentation_params['limite_var']}")
                print(f"limite_media={segmentation_params['limite_media']}")
                print(f"referencia_media={segmentation_params['referencia_media']}")    
                mascara_segmentada = aplicar_divisao_e_fusao(
                    pixel_array,segmentation_params['limite_var'],
                    segmentation_params['limite_media'],segmentation_params['referencia_media'])  
                todos_os_contornos,contornos_validos=remove_fundo(mascara_segmentada,postprocessing_params['area_minima'])
        
        elif method == "crescimento_regioes_fora":            
            imagem_segmentada_8bits_invertida = crescimento_regioes_fora(imagem_hu)
            todos_os_contornos,contornos_validos=remove_fundo(imagem_segmentada_8bits_invertida)

        elif method == "otsu":
            mascara_segmentada = aplicar_otsu(pixel_array)
            todos_os_contornos,contornos_validos=remove_fundo(mascara_segmentada)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Método de segmentação inválido. Use 'watershed' ou 'lim_media_mov' ou 'lim_multipla' ou 'lim_prop_locais', 'sauvola', 'otsu' ou 'divisao_e_fusao'.",
            )
        
        if segmentation_params or method == 'otsu' or method == 'crescimento_regioes_fora' or method=='segmentation':
            pixel_array=imagem_para_base64(pixel_array)
            print(pixel_array)
            print("Todos os contornos:", todos_os_contornos)
            print("Contornos válidos:", contornos_validos)
            return JSONResponse({"imagem_pre_processada":pixel_array, "todos_os_contornos":todos_os_contornos, "contornos_validos":contornos_validos})
        
        return JSONResponse ({"imagem_pre_processada":{},"todos_os_contornos":{},"contornos_validos":{}})
    except Exception as e:
        stack_trace = traceback.format_exc()  # Captura a stack trace como string
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar o arquivo DICOM: {str(e)}\n\nStack trace:\n{stack_trace}"
        )
