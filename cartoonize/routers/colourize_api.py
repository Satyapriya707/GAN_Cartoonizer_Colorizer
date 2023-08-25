from fastapi import FastAPI, Query, APIRouter, status
from fastapi.responses import JSONResponse
from typing import Optional, List, final
import shutil
import os

import requests
from model.errorMessage import Message
from cartoon_gan.inference import output_video_dir

router = APIRouter(tags = ["Video Colourization API Call"])

@router.post("/colourizeVideoAPI", summary="Colourize Video", status_code=status.HTTP_202_ACCEPTED, responses={404: {"model": Message}})
def colourizeVideoAPIFunction(videoDirectory: str = Query(..., max_length=100),
                    videoName: str = Query(..., max_length=100),
                    extension: Optional[str] = Query("mp4", max_length=20),
                    directoryToSaveTo: Optional[str] = Query(None, max_length=100),
                    gpuID: Optional[int] = Query(0)):
    url = "http://localhost:8080"
    vid_path = os.path.join(videoDirectory, f"{videoName}.{extension}")
    if not os.path.isfile(vid_path):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The video in path {vid_path} does not exist"})
    output_dir_final = output_video_dir
    if directoryToSaveTo:
        output_dir_final = os.path.join(output_video_dir, directoryToSaveTo)
    if not os.path.isdir(output_dir_final):
        # os.mkdir(output_dir_final)
        os.makedirs(output_dir_final)
    param = {
        "videoDirectory": videoDirectory,
        "videoName": videoName,
        "extension": extension,
        "directoryToSaveTo": os.path.abspath(output_dir_final)
    }
    val = requests.post(f"{url}/colourizeVideo", params=param)
    if val:
        return "Video colourized"
    return "Failed"
    