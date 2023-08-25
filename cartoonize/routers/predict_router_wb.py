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
from utils.check_video import vid_input, vid_output_wb, remove_dirs
from subprocess import check_output
from utils.zip_response import zipfiles

from utils.model_style_map import model_dict
from subprocess import check_output
from utils.add_audio import audio_add
import shutil

router = APIRouter(tags = ["W-B Cartoonization Prediction Using Pytorch"])

@router.post("/predictWhiteBoxPytorch", response_class=FileResponse, summary="W-B Cartoonization Prediction on Single Image", status_code=status.HTTP_202_ACCEPTED, responses={404: {"model": Message}})
def imagePrediction(image: UploadFile,
                    imageDirectory: Optional[str] = Query(None, max_length=100),
                    style: dropdownChoicesSingleImageWB = Form(dropdownChoicesSingleImageWB.type_3),
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
    shutil.copy(img, f"wb_cartoonization/asset/{imageName}")
    img = f"wb_cartoonization/asset/{imageName}"
    img_to_cli = f"asset/{imageName}"
    if useGPU:
        dev = "cuda"
    else:
        dev = "cpu"
    if maximumSize:
        inp = f"python scripts/whiteboxgan.py --stage infer --ckpt={model_dict[style]} --extra=image_path:{img_to_cli},device:{dev},load_size:{maximumSize}"
    else:
        inp = f"python scripts/whiteboxgan.py --stage infer --ckpt={model_dict[style]} --extra=image_path:{img_to_cli},device:{dev}"
    check_output(f"cd wb_cartoonization && {inp}", shell=True)
    os.remove(img)
    final_name_path = f'{".".join(img.split(".")[:-1])}_out.{ext}'
    if not maximumSize:
        maximumSize = 0
    final_changed_name = f'{".".join(imageName.split(".")[:-1])}_{style.lower()}_{str(maximumSize)}.{ext}'
    final_saved_path = os.path.join(output_dir_final, final_changed_name)
    shutil.move(final_name_path, final_saved_path)
    return FileResponse(final_saved_path, headers={"location": f"The cartoonized image can be found at - {final_saved_path}"})


@router.post("/predictBatchWhiteBoxPytorch", response_model=str, summary="W-B Cartoonization Prediction on Batch of Images", status_code=status.HTTP_202_ACCEPTED, responses={404: {"model": Message}})
def imagePredictionBatch(imageDirectory: str = Query(..., example=r"D:\Projects\cartoon_gan\test\input_images", max_length=100),
                    styleType: dropdownChoicesWB = Form(dropdownChoicesWB.type_3),
                    maximumSize: Optional[int] = Query(None),
                    directoryToSaveTo: Optional[str] = Query(None, max_length=100),
                    useGPU: Optional[bool] = Query(True),
                    batchSize: Optional[int] = Query(None)):

    if styleType == "All":
        style_list = ["Type-0", "Type-1", "Type-2", "Type-3"]
    else:
        style_list = [styleType]
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
    len_style = len(style_list)
    shutil.copytree(imageDirectory, f"wb_cartoonization/asset/{dir_name}")
    if useGPU:
        dev = "cuda"
    else:
        dev = "cpu"
    for style_count, style in enumerate(style_list):
        print(f"Processing for style {style_count+1}/{len_style} : {style}")
        if not batchSize:
            batchSize = 16
        if maximumSize:
            inp = f"python scripts/whiteboxgan.py --stage infer --ckpt={model_dict[style]} --extra=image_path:asset/{dir_name},device:{dev},batch_size:{batchSize},load_size:{maximumSize}"
        else:    
            inp = f"python scripts/whiteboxgan.py --stage infer --ckpt={model_dict[style]} --extra=image_path:asset/{dir_name},batch_size:{batchSize},device:{dev}"
        check_output(f"cd wb_cartoonization && {inp}", shell=True)
        for file in os.listdir(f"wb_cartoonization/asset/{dir_name}_out"):
            ext = file.split(".")[-1]
            maxS = maximumSize if maximumSize else 0
            final_name = f'{".".join(file.split(".")[:-1])}_{style.lower()}_{str(maxS)}.{ext}'
            shutil.move(os.path.join(f"wb_cartoonization/asset/{dir_name}_out", file), os.path.join(output_dir_final, final_name))
    shutil.rmtree(f"wb_cartoonization/asset/{dir_name}")
    shutil.rmtree(f"wb_cartoonization/asset/{dir_name}_out")
    return f"All the cartoonized images can be found at - {output_dir_final}"


@router.post("/predictOnVideoWhiteBoxPytorch", summary="W-B Cartoonization Prediction on Single Video", status_code=status.HTTP_202_ACCEPTED, responses={404: {"model": Message}})
def imagePredictionVideo(videoFile: UploadFile,
                    videoDirectory: Optional[str] = Query(None, max_length=100),
                    styleType: dropdownChoicesWB = Form(dropdownChoicesWB.type_3),
                    maximumSize: Optional[int] = Query(None),
                    fps: Optional[int] = Query(30),
                    downloadZip: Optional[bool] = Query(False),
                    directoryToSaveTo: Optional[str] = Query(None, max_length=100),
                    useGPU: Optional[bool] = Query(True),
                    batchSize: Optional[int] = Query(None)):
    
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
    shutil.copy(vid, f"wb_cartoonization/asset/{videoName}")
    videoDirectory = "wb_cartoonization/asset"
    temp_dir = vid_input(videoNameOnly, extVid, fr'{videoDirectory}', fps)
    if styleType == "All":
        style_list = ["Type-0", "Type-1", "Type-2", "Type-3"]
    else:
        style_list = [styleType]   
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
    len_style = len(style_list)
    output_path = []
    for style_count, style in enumerate(style_list):
        print(f"Processing for style {style_count+1}/{len_style} : {style}")
        if not batchSize:
            batchSize = 16
        if maximumSize:
            inp = f"python scripts/whiteboxgan.py --stage infer --ckpt={model_dict[style]} --extra=image_path:asset/{temp_dir},device:{dev},batch_size:{batchSize},load_size:{maximumSize}"
        else:    
            inp = f"python scripts/whiteboxgan.py --stage infer --ckpt={model_dict[style]} --extra=image_path:asset/{temp_dir},batch_size:{batchSize},device:{dev}"
        check_output(f"cd wb_cartoonization && {inp}", shell=True)
        if not maximumSize:
            maximumSize = 0
        vid_output_wb(videoNameOnly, extVid, videoDirectory, fr'{output_dir_final}', style.lower(), maximumSize, fps)
        audio_add(vid, os.path.join(output_dir_final, f"{videoNameOnly}_{style.lower()}_{maximumSize}.{extVid}"))
        output_path.append(os.path.join(output_dir_final, f"{videoNameOnly}_{style.lower()}_{maximumSize}.{extVid}"))
        shutil.rmtree(os.path.join(videoDirectory, f"{temp_dir}_out"))
    shutil.rmtree(os.path.join(videoDirectory, temp_dir))
    os.remove(f"wb_cartoonization/asset/{videoName}")
    if downloadZip:
        return zipfiles(output_path)
    else:
        if len(style_list) == 1:
            if extVid == "gif":
                return FileResponse(os.path.join(output_dir_final, f"{videoNameOnly}_{style.lower()}_{maximumSize}.{extVid}"), headers={"location": f"The cartoonized images can be found at - '{output_dir_final}' with name '{videoNameOnly}_{style.lower()}_{maximumSize}.{extVid}'"})    
            else:
                return f"The cartoonized video can be found at - '{output_dir_final}' with name '{videoNameOnly}_{style.lower()}_{maximumSize}.{extVid}'"
        else:
            return f"The cartoonized videos can be found at - '{output_dir_final}' with names '{videoNameOnly}_<style_name>_{maximumSize}.{extVid}'" 
          


@router.post("/predictOnMultipleVideosWhiteBoxPytorch", summary="W-B Cartoonization Prediction on Multiple Videos", status_code=status.HTTP_202_ACCEPTED, responses={404: {"model": Message}})
def imagePredictionMultipleVideos(videoDirectory: Optional[str] = Query(None, max_length=100),
                    styleType: dropdownChoicesWB = Form(dropdownChoicesWB.type_3),
                    maximumSize: Optional[int] = Query(None),
                    fps: Optional[int] = Query(30),
                    downloadZip: Optional[bool] = Query(False),
                    directoryToSaveTo: Optional[str] = Query(None, max_length=100),
                    useGPU: Optional[bool] = Query(True),
                    batchSize: Optional[int] = Query(None)):
    
    if not videoDirectory:
        videoDirectory = input_video_dir
    if not os.path.isdir(videoDirectory):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The directory path {videoDirectory} does not exist"})
    videoList = os.listdir(videoDirectory)
    len_videoList = len(videoList)
    output_path = []

    if styleType == "All":
        style_list = ["Type-0", "Type-1", "Type-2", "Type-3"]
    else:
        style_list = [styleType]  

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
    vid_dir_new = f"wb_cartoonization/asset/{dir_name}"
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
        len_style = len(style_list)
        for style_count, style in enumerate(style_list):
            print(f"Processing for style {style_count+1}/{len_style} : {style}")
            if not batchSize:
                batchSize = 16
            if maximumSize:
                inp = f"python scripts/whiteboxgan.py --stage infer --ckpt={model_dict[style]} --extra=image_path:asset/{dir_name}/{temp_dir},device:{dev},batch_size:{batchSize},load_size:{maximumSize}"
            else:    
                inp = f"python scripts/whiteboxgan.py --stage infer --ckpt={model_dict[style]} --extra=image_path:asset/{dir_name}/{temp_dir},batch_size:{batchSize},device:{dev}"
            check_output(f"cd wb_cartoonization && {inp}", shell=True)
            if not maximumSize:
                maximumSize = 0
            vid_output_wb(videoNameOnly, extVid, videoDirectory, fr'{output_dir_final}', style.lower(), maximumSize, fps)
            audio_add(vid, os.path.join(output_dir_final, f"{videoNameOnly}_{style.lower()}_{maximumSize}.{extVid}"))
            output_path.append(os.path.join(output_dir_final, f"{videoNameOnly}_{style.lower()}_{maximumSize}.{extVid}"))
            shutil.rmtree(os.path.join(videoDirectory, f"{temp_dir}_out"))
        shutil.rmtree(imageDirectory)
        os.remove(vid)
    shutil.rmtree(videoDirectory)
    if downloadZip:
        return zipfiles(output_path)
    else:
        return f"The cartoonized videos can be found at - {output_dir_final}"          

