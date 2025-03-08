from typing import List, Tuple

def segment_image(pixel_array) -> List[Tuple[float, float]]:
    """
    Funcao experimental para testar a api localmente (a partir daqui sera implementada a funcao de segmentacao)
    
    """
    points = []
    
    height, width = pixel_array.shape
    for i in range(0, height, int(height / 5)):  
        for j in range(0, width, int(width / 5)):  
            points.append((float(i), float(j)))
    
    return points

