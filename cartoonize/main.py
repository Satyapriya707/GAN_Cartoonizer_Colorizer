import os
from fastapi import FastAPI
import uvicorn
from routers import predict_router_cg, predict_router_wb , predict_router_wb_tf2, video_processing, colourize_api
from cartoon_gan.inference import input_dir, output_dir, input_video_dir, output_video_dir

app = FastAPI()

app.include_router(predict_router_cg.router)
app.include_router(predict_router_wb.router)
app.include_router(predict_router_wb_tf2.router)
app.include_router(video_processing.router)
app.include_router(colourize_api.router)


if __name__ == '__main__':
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_video_dir, exist_ok=True)
    os.makedirs(output_video_dir, exist_ok=True)
    uvicorn.run("main:app", port=8000, host='127.0.0.1', reload=True)