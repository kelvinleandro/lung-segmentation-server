from typing import Dict, Any


def converte_param_preprocess(preprocessing_params: Dict[str, str]) -> Dict[str, Any]:
    # Converte para inteiro (se o valor não for fornecido, usa o valor padrão 5)
    if "tamanho_kernel" in preprocessing_params:
        preprocessing_params["tamanho_kernel"] = int(
            preprocessing_params["tamanho_kernel"]
        )

    # Converte para float (se o valor não for fornecido, usa o valor padrão 0)
    if "sigma" in preprocessing_params:
        preprocessing_params["sigma"] = float(preprocessing_params["sigma"])

    return preprocessing_params


from typing import Dict, Any


def converter_parametros_para_tipos(
    segmentation_params: Dict[str, str],
) -> Dict[str, Any]:
    # Converte para inteiro
    int_params = [
        "limite_var",
        "limite_media",
        "referencia_media",
        "tamanho_kernel",
        "iteracoes_morfologia",
        "iteracoes_dilatacao",
        "tamanho_janela",
        "n",
        "delta_limiar",
    ]
    for param in int_params:
        if param in segmentation_params:
            segmentation_params[param] = int(segmentation_params[param])

    # Converte para float
    float_params = ["fator_dist_transform", "b", "a", "k", "limiar"]
    for param in float_params:
        if param in segmentation_params:
            segmentation_params[param] = float(segmentation_params[param])

    # Converte para tupla (separando por vírgula)
    tuple_params = [
        "lim_hiperaeradas",
        "lim_normalmente_aeradas",
        "lim_pouco_aeradas",
        "lim_nao_aeradas",
        "lim_osso",
    ]
    for param in tuple_params:
        if param in segmentation_params:
            segmentation_params[param] = tuple(
                map(int, segmentation_params[param].split(","))
            )

    # Se necessário, você pode adicionar outros parâmetros aqui, como listas ou booleans.

    return segmentation_params
