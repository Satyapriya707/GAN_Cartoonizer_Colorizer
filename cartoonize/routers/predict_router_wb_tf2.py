from tkinter.ttk import Style
from typing import Optional, List
from wsgiref import headers
from fastapi import FastAPI, Query, APIRouter, status, Form, UploadFile
from pydantic import BaseModel, DirectoryPath
import os
import uvicorn
from model.errorMessage import Message
from model.models import dropdownChoicesWB, dropdownChoicesSingleImageWB
from fastapi.responses import JSONResponse

from cartoon_gan.inference import input_dir, output_dir, input_video_dir, output_video_dir
from fastapi.responses import FileResponse
import time
from utils.check_video import vid_input, vid_output_wb_tf, remove_dirs
from subprocess import check_output
from utils.zip_response import zipfiles

from utils.model_style_map import model_dict
from subprocess import check_output
from utils.add_audio import audio_add
import shutil

router = APIRouter(tags = ["W-B Cartoonization Prediction Using Tensorflow"])

@router.post("/predictWhiteBoxTF", response_class=FileResponse, summary="W-B Cartoonization Prediction on Single Image", status_code=status.HTTP_202_ACCEPTED, responses={404: {"model": Message}})
def imagePrediction(image: UploadFile,
                    imageDirectory: Optional[str] = Query(None, max_length=100),
                    maximumSize: Optional[int] = Query(None),
                    directoryToSaveTo: Optional[str] = Query(None, max_length=100),
                    useGPU: Optional[bool] = Query(True)):

    if not imageDirectory:
        imageDirectory = input_dir
    if not os.path.isdir(imageDirectory):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The directory path {imageDirectory} does not exist"})
    imageName = image.filename
    ext = imageName.split(".")[-1]
    img = os.path.join(imageDirectory, imageName)
    if not os.path.isfile(img):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The image path {img} does not exist"})
    output_dir_final = output_dir
    if directoryToSaveTo:
        output_dir_final = os.path.join(output_dir, directoryToSaveTo)
    if not os.path.isdir(output_dir_final):
        # os.mkdir(output_dir_final)
        os.makedirs(output_dir_final)
    shutil.copy(img, f"white_box_cartoonizer/asset/{imageName}")
    img = f"white_box_cartoonizer/asset/{imageName}"
    if useGPU:
        dev = "gpu"
    else:
        dev = "cpu"
    if maximumSize:
        inp = f"python white_box_cartoonizer/cartoonize.py --{dev} --image  --path={img} --maxSize={maximumSize}"
    else:
        inp = f"python white_box_cartoonizer/cartoonize.py --{dev} --image  --path={img}"
    check_output(f"{inp}", shell=True)
    os.remove(img)
    final_name_path = f'{".".join(img.split(".")[:-1])}_out.{ext}'
    if not maximumSize:
        maximumSize = 0
    final_changed_name = f'{".".join(imageName.split(".")[:-1])}_{str(maximumSize)}.{ext}'
    final_saved_path = os.path.join(output_dir_final, final_changed_name)
    shutil.move(final_name_path, final_saved_path)
    return FileResponse(final_saved_path, headers={"location": f"The cartoonized image can be found at - {final_saved_path}"})


@router.post("/predictBatchWhiteBoxTF", response_model=str, summary="W-B Cartoonization Prediction on Batch of Images", status_code=status.HTTP_202_ACCEPTED, responses={404: {"model": Message}})
def imagePredictionBatch(imageDirectory: str = Query(..., example=r"D:\Projects\cartoon_gan\test\input_images", max_length=100),
                    maximumSize: Optional[int] = Query(None),
                    directoryToSaveTo: Optional[str] = Query(None, max_length=100),
                    useGPU: Optional[bool] = Query(True)):

    if not os.path.isdir(imageDirectory):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The directory path {imageDirectory} does not exist"})
    # dir_name = imageDirectory.split("\\")[-1]  
    dir_name = os.path.basename(imageDirectory) 
    output_dir_final = output_dir
    if directoryToSaveTo:
        output_dir_final = os.path.join(output_dir, directoryToSaveTo)
    if not os.path.isdir(output_dir_final):
        # os.mkdir(output_dir_final)
        os.makedirs(output_dir_final)
    if not os.path.isdir(f"white_box_cartoonizer/asset/{dir_name}"):
        shutil.copytree(imageDirectory, f"white_box_cartoonizer/asset/{dir_name}")
    if useGPU:
        dev = "cuda"
    else:
        dev = "cpu"
    if maximumSize:
        inp = f"python white_box_cartoonizer/cartoonize.py --{dev} --batch  --path=white_box_cartoonizer/asset/{dir_name} --maxSize={maximumSize}"
    else:    
        inp = f"python white_box_cartoonizer/cartoonize.py --{dev} --batch  --path=white_box_cartoonizer/asset/{dir_name}"
    check_output(f"{inp}", shell=True)
    maxS = maximumSize if maximumSize else 0
    for file in os.listdir(f"white_box_cartoonizer/asset/{dir_name}_out"):
        ext = file.split(".")[-1]
        final_name = f'{".".join(file.split(".")[:-1])}_{str(maxS)}.{ext}'
        shutil.move(os.path.join(f"white_box_cartoonizer/asset/{dir_name}_out", file), os.path.join(output_dir_final, final_name))
    shutil.rmtree(f"white_box_cartoonizer/asset/{dir_name}")
    shutil.rmtree(f"white_box_cartoonizer/asset/{dir_name}_out")
    return f"All the cartoonized images can be found at - {output_dir_final}"


@router.post("/predictOnVideoWhiteBoxTF", summary="W-B Cartoonization Prediction on Single Video", status_code=status.HTTP_202_ACCEPTED, responses={404: {"model": Message}})
def imagePredictionVideo(videoFile: UploadFile,
                    videoDirectory: Optional[str] = Query(None, max_length=100),
                    maximumSize: Optional[int] = Query(None),
                    fps: Optional[int] = Query(30),
                    downloadZip: Optional[bool] = Query(False),
                    directoryToSaveTo: Optional[str] = Query(None, max_length=100),
                    useGPU: Optional[bool] = Query(True)):
    
    if not videoDirectory:
        videoDirectory = input_video_dir
    if not os.path.isdir(videoDirectory):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The directory path {videoDirectory} does not exist"})
    videoName = videoFile.filename
    vid = os.path.join(videoDirectory, videoName)
    if not os.path.isfile(vid):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The video file path {vid} does not exist"})
    extVid = videoName.split(".")[-1]
    videoNameOnly = ".".join(videoName.split(".")[:-1])
    shutil.copy(vid, f"white_box_cartoonizer/asset/{videoName}")
    videoDirectory = "white_box_cartoonizer/asset"
    temp_dir = vid_input(videoNameOnly, extVid, fr'{videoDirectory}', fps)
    imageDirectory = os.path.join(videoDirectory, temp_dir)
    if not os.path.isdir(imageDirectory):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The directory path {imageDirectory} does not exist"})
    output_dir_final = output_video_dir
    if directoryToSaveTo:
        output_dir_final = os.path.join(output_video_dir, directoryToSaveTo)
    if not os.path.isdir(output_dir_final):
        # os.mkdir(output_dir_final)
        os.makedirs(output_dir_final)
    if useGPU:
        dev = "cuda"
    else:
        dev = "cpu"
    output_path = []
    if maximumSize:
        inp = f"python white_box_cartoonizer/cartoonize.py --{dev} --batch  --path=white_box_cartoonizer/asset/{temp_dir} --maxSize={maximumSize}"
    else:    
        inp = f"python white_box_cartoonizer/cartoonize.py --{dev} --batch  --path=white_box_cartoonizer/asset/{temp_dir}"
    check_output(f"{inp}", shell=True)
    if not maximumSize:
        maximumSize = 0
    vid_output_wb_tf(videoNameOnly, extVid, videoDirectory, fr'{output_dir_final}', maximumSize, fps)
    audio_add(vid, os.path.join(output_dir_final, f"{videoNameOnly}_{maximumSize}.{extVid}"))
    output_path.append(os.path.join(output_dir_final, f"{videoNameOnly}_{maximumSize}.{extVid}"))
    shutil.rmtree(os.path.join(videoDirectory, f"{temp_dir}_out"))
    shutil.rmtree(os.path.join(videoDirectory, temp_dir))
    os.remove(f"white_box_cartoonizer/asset/{videoName}")
    if downloadZip:
        return zipfiles(output_path, os.path.join(output_dir_final, f"{videoNameOnly}_{maximumSize}.{extVid}"))
    else:
        if extVid == "gif":
            return FileResponse(os.path.join(output_dir_final, f"{videoNameOnly}_{maximumSize}.{extVid}"), headers={"location": f"The cartoonized images can be found at - '{output_dir_final}' with name '{videoNameOnly}_{maximumSize}.{extVid}'"})    
        else:
            return f"The cartoonized video can be found at - '{output_dir_final}' with name '{videoNameOnly}_{maximumSize}.{extVid}'"
        


@router.post("/predictOnMultipleVideosWhiteBoxTF", summary="W-B Cartoonization Prediction on Multiple Videos", status_code=status.HTTP_202_ACCEPTED, responses={404: {"model": Message}})
def imagePredictionMultipleVideos(videoDirectory: Optional[str] = Query(None, max_length=100),
                    maximumSize: Optional[int] = Query(None),
                    fps: Optional[int] = Query(30),
                    downloadZip: Optional[bool] = Query(False),
                    directoryToSaveTo: Optional[str] = Query(None, max_length=100),
                    useGPU: Optional[bool] = Query(True)):
    
    if not videoDirectory:
        videoDirectory = input_video_dir
    if not os.path.isdir(videoDirectory):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The directory path {videoDirectory} does not exist"})
    videoList = os.listdir(videoDirectory)
    len_videoList = len(videoList)
    output_path = []

    output_dir_final = output_video_dir
    if directoryToSaveTo:
        output_dir_final = os.path.join(output_video_dir, directoryToSaveTo)
    if not os.path.isdir(output_dir_final):
        # os.mkdir(output_dir_final)
        os.makedirs(output_dir_final)
    if useGPU:
        dev = "cuda"
    else:
        dev = "cpu"
    dir_name = os.path.basename(videoDirectory)
    vid_dir_new = f"white_box_cartoonizer/asset/{dir_name}"
    shutil.copytree(videoDirectory, vid_dir_new)
    videoDirectory = vid_dir_new
    for vidCount, videoName in enumerate(videoList):
        print(f"Processing for video {vidCount+1}/{len_videoList} : {videoName}")
        vid = os.path.join(videoDirectory, videoName)
        if not os.path.isfile(vid):
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The video file path {vid} does not exist"})
        extVid = videoName.split(".")[-1]
        videoNameOnly = ".".join(videoName.split(".")[:-1])
        temp_dir = vid_input(videoNameOnly, extVid, fr'{videoDirectory}', fps)
        imageDirectory = os.path.join(videoDirectory, temp_dir)
        if not os.path.isdir(imageDirectory):
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The directory path {imageDirectory} does not exist"})
        t0 = time.time()
        if maximumSize:
            inp = f"python white_box_cartoonizer/cartoonize.py --{dev} --batch  --path={videoDirectory}/{temp_dir} --maxSize={maximumSize}"
        else:    
            inp = f"python white_box_cartoonizer/cartoonize.py --{dev} --batch  --path={videoDirectory}/{temp_dir}"
        check_output(f"{inp}", shell=True)
        if not maximumSize:
            maximumSize = 0
        vid_output_wb_tf(videoNameOnly, extVid, videoDirectory, fr'{output_dir_final}', maximumSize, fps)
        audio_add(vid, os.path.join(output_dir_final, f"{videoNameOnly}_{maximumSize}.{extVid}"))
        output_path.append(os.path.join(output_dir_final, f"{videoNameOnly}_{maximumSize}.{extVid}"))
        shutil.rmtree(os.path.join(videoDirectory, f"{temp_dir}_out"))
        shutil.rmtree(imageDirectory)
        os.remove(vid)
    shutil.rmtree(videoDirectory)
    if downloadZip:
        return zipfiles(output_path)
    else:
        return f"The cartoonized videos can be found at - {output_dir_final}"   