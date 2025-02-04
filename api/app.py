from flask import Flask, request, send_file
from flask_cors import CORS  
import pydicom
import numpy as np
import io

app = Flask(__name__)
CORS(app,origins=["http://localhost:8000"])  

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if not file:
        return {"error": "Nenhum arquivo enviado"}, 400

    # Carregar imagem DICOM
    dicom_data = pydicom.dcmread(file)

    # Simples processamento (inversão de intensidade como exemplo)
    dicom_data.PixelData = (np.max(dicom_data.pixel_array) - dicom_data.pixel_array).tobytes()
    
    # Salvar em memória
    output = io.BytesIO()
    dicom_data.save_as(output)
    output.seek(0)
    output.seek(128)  # Pula os primeiros 128 bytes
    signature = output.read(4)  # Lê os 4 bytes seguintes

    if signature == b"DICM":
        print("Assinatura DICOM válida encontrada!")
    else:
        print("Erro: arquivo salvo no buffer pode estar corrompido.")
    
    output.seek(0)
    return send_file(output, mimetype='application/dicom', as_attachment=True, download_name="processed.dcm")
    
if __name__ == '__main__':
    app.run(debug=True)
