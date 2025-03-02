import asyncio
import time

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.status import HTTP_504_GATEWAY_TIMEOUT

from api.api import api_router
from schemas.healthcheck import HealthCheck

REQUEST_TIMEOUT_ERROR = 300

# Initialize FastAPI app
app = FastAPI(
    title="API - Projeto de PDI",
    openapi_url="/api/v1/openapi.json",
    openapi_tags=[
        {"name": "Healthcheck", "description": "Healthcheck endpoint for the API."},
        {
            "name": "Segmentation",
            "description": "Segmentation endpoint for DICOM images.",
        },
    ],
)


# Middleware for handling request timeouts
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        start_time = time.time()
        response = await asyncio.wait_for(
            call_next(request), timeout=REQUEST_TIMEOUT_ERROR
        )
        return response
    except asyncio.TimeoutError:
        process_time = time.time() - start_time
        return JSONResponse(
            {"detail": f"Request processing time exceeded limit ({process_time})"},
            status_code=HTTP_504_GATEWAY_TIMEOUT,
        )


# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# healthcheck endpoint
@app.get("/", response_model=HealthCheck, tags=["Healthcheck"])
async def healthcheck(request: Request):
    return {"message": "OK"}


# API router
app.include_router(api_router, prefix="/api")
