# Lung Segmentation Server

## Overview

This repository contains the backend of the lung segmentation project, responsible for processing the DICOM files uploaded by the [frontend](https://github.com/kelvinleandro/lung-segmentation-client), running the segmentation algorithms, and returning the processed results.

The backend was developed using **FastAPI**. To optimize mathematical and image operations, the **NumPy** and **Numba** libraries were used, enabling vectorized calculations and parallel processing.

## Technologies Used

- **API**: FastAPI
- **Numerical and Vectorized Processing**: NumPy, Numba
- **Parallel Execution and Optimization**: Numba
- **Reading and Processing DICOM Images**: Pydicom

## Features

- ✅ Processing of DICOM files (.dcm)
- ✅ Implementation of segmentation methods (MCA Crisp, Otsu, Watershed, Sauvola, Division and Fusion, Seed Growth in Region Outside the Lung, Moving Average Threshold, Multiple Threshold, and Local Properties Threshold)
- ✅ Optimization of algorithms using Numba
- ✅ REST API for communication with the frontend
- ✅ Generation of contours and segmented image

## Installation and Configuration

### 1. Create and Activate Virtual Environment (venv)

It is recommended to use a virtual environment to manage dependencies:

```
python -m venv .venv
```

Activate the venv:

```
Linux/Mac:
source .venv/bin/activate

Windows:
.venv\Scripts\activate
```

### Install Dependencies

```
pip install -r requirements.txt
```

### Run the Local Server

```
cd app
make run-dev
```
