# Backend - Projeto Final de PDI

## Visão Geral

Este repositório contém o backend do projeto de segmentação pulmonar, responsável por processar os arquivos DICOM enviados pelo frontend, executar os algoritmos de segmentação e retornar os resultados processados.

O backend foi desenvolvido utilizando **FastAPI**. Para otimização das operações matemáticas e de imagem, foram utilizadas as bibliotecas **NumPy** e **Numba**, permitindo cálculos vetorizados e processamento paralelo.

## Tecnologias Utilizadas

- **API**: FastAPI
- **Processamento Numérico e Vetorizado**: NumPy, Numba
- **Execução Paralela e Otimização**: Numba
- **Leitura e Processamento de Imagens DICOM**: Pydicom

## Funcionalidades

- ✅ Processamento de arquivos DICOM (.dcm)
- ✅ Implementação de métodos de segmentação (MCA Crisp, Otsu, Watershed, Sauvola, Divisão e Fusão, Crescimento de semente em região fora do pulmão, Limite Média Móvel, Limite Múltiplo e  Limite Propriedades Locais)
- ✅ Otimização de algoritmos utilizando Numba
- ✅ API REST para comunicação com o frontend
- ✅ Geração de contornos e imagem segmentada

## Instalação e Configuração

### 1. Criar e Ativar Ambiente Virtual (venv)

É recomendado o uso de um ambiente virtual para gerenciar dependências:

```bash
python -m venv .venv
```

Ative o venv:

```bash
Linux/Mac:
source .venv/bin/activate

Windows:
.venv\Scripts\activate

```
### Instalação das dependências

```bash
pip install -r requirements.txt
```

### Rodar o Servidor local 

```cd app
make run-dev
```
