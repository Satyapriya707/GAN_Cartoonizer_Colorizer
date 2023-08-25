from typing import Optional, List
from fastapi import FastAPI, Query, APIRouter, status, Form, UploadFile
from pydantic import BaseModel, DirectoryPath
import os
import uvicorn
from model.errorMessage import Message
from fastapi.responses import JSONResponse
from subprocess import check_output
from utils.add_audio import audio_add

router = APIRouter(tags = ["Utils To Deal With Video And Audio"])

@router.post("/createVideo", summary="Generate Video From Images", status_code=status.HTTP_202_ACCEPTED, responses={404: {"model": Message}})
def createVideo(imageDirectory: str = Query(..., max_length=100),
                    fps: Optional[int] = Query(None),
                    directoryToSaveTo: Optional[str] = Query(None, max_length=100),
                    videoName: str = Query(..., max_length=100),
                    extension: Optional[str] = Query("mp4", max_length=20),
                    imageNamePattern: str = Query(..., max_length=100)):

    output_dir = "output_video"
    if not os.path.isdir(imageDirectory):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The directory path {imageDirectory} does not exist"})   
    if directoryToSaveTo:
        output_dir = os.path.join(output_dir, directoryToSaveTo)
    if not os.path.isdir(output_dir):
        # os.mkdir(output_dir)
        os.makedirs(output_dir)
    if extension == "gif":
        out = f'ffmpeg -r {fps} -i "{imageDirectory}/${videoName}%3d{imageNamePattern}.bmp" {output_dir}/{videoName}.{extension}'
    else:
        out = f'ffmpeg -r {fps} -i "{imageDirectory}/${videoName}%3d{imageNamePattern}.bmp" -c:v libx264 -pix_fmt yuv420p {output_dir}/{videoName}.{extension}'        
    check_output(out, shell=True)
    return "Done"


@router.post("/addAudio", summary="Add Audio To Video", status_code=status.HTTP_202_ACCEPTED, responses={404: {"model": Message}})
def createVideo(videoWithAudioPath: str = Query(..., max_length=100),
                    videoWithoutAudioPath: str = Query(..., max_length=100)):
    val = audio_add(videoWithAudioPath, videoWithoutAudioPath)
    if val:
        return "Audio has been added to the video"
    else:
        return "The video does not contain any audio"


@router.post("/createFrames", summary="Generate Images From Video", status_code=status.HTTP_202_ACCEPTED, responses={404: {"model": Message}})
def createVideo(videoDirectory: str = Query(..., max_length=100),
                    fps: Optional[int] = Query(None),
                    directoryToSaveTo: Optional[str] = Query(None, max_length=100),
                    videoName: str = Query(..., max_length=100),
                    extension: Optional[str] = Query("mp4", max_length=20)):
    
    output_dir = "output_images"
    if not os.path.isdir(videoDirectory):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The directory path {videoDirectory} does not exist"})  
    vid_path = os.path.join(videoDirectory, f"{videoName}.{extension}")
    if not os.path.isfile(vid_path):
          return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The video path {vid_path} does not exist"})  
    if directoryToSaveTo:
        output_dir = os.path.join(output_dir, directoryToSaveTo)
    if not os.path.isdir(output_dir):
        # os.mkdir(output_dir)
        os.makedirs(output_dir)
    inp = f'ffmpeg -i "{vid_path}" -r {fps} {output_dir}/${videoName}%03d.bmp'
    check_output(inp, shell=True)
    return "Done"
