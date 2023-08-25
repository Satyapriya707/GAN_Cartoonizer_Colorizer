from tkinter.ttk import Style
from typing import Optional, List
from wsgiref import headers
from fastapi import FastAPI, Query, APIRouter, status, Form, UploadFile
from pydantic import BaseModel, DirectoryPath
import os
import uvicorn
from model.errorMessage import Message
from model.models import dropdownChoices, dropdownChoicesSingleImage
from fastapi.responses import JSONResponse

from cartoon_gan.inference import transform, load_model, input_dir, output_dir, input_video_dir, output_video_dir
from fastapi.responses import FileResponse
import time
from utils.check_video import vid_input, vid_output, remove_dirs
from utils.add_audio import audio_add
from subprocess import check_output
from utils.zip_response import zipfiles

router = APIRouter(tags = ["Cartoon GAN Prediction"])

@router.post("/predict", response_class=FileResponse, summary="Cartoon GAN Prediction on Single Image", status_code=status.HTTP_202_ACCEPTED, responses={404: {"model": Message}})
def imagePrediction(image: UploadFile,
                    imageDirectory: Optional[str] = Query(None, max_length=100),
                    style: dropdownChoicesSingleImage = Form(dropdownChoicesSingleImage.hosoda),
                    maximumSize: Optional[int] = Query(None),
                    directoryToSaveTo: Optional[str] = Query(None, max_length=100),
                    useGPU: Optional[bool] = Query(True)):

    if not imageDirectory:
        imageDirectory = input_dir
    if not os.path.isdir(imageDirectory):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The directory path {imageDirectory} does not exist"})
    model = load_model(style, useGPU)
    imageName = image.filename
    ext = imageName.split(".")[-1]
    img = os.path.join(imageDirectory, imageName)
    if not os.path.isfile(img):
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The image path {img} does not exist"})
    output_dir_final = output_dir
    if directoryToSaveTo:
        output_dir_final = os.path.join(output_dir, directoryToSaveTo)
    if not os.path.isdir(output_dir_final):
        os.makedirs(output_dir_final)
    output_img, maxS = transform(model, style, img, maximumSize, useGPU)
    if not maximumSize:
        maximumSize = maxS
    img_path = os.path.join(output_dir_final ,".".join(imageName.split(".")[:-1]) + '_' + style.lower() + "_" + str(maximumSize) + '.' + ext)
    output_img.save(img_path)
    return FileResponse(img_path, headers={"location": f"The cartoonized image can be found at - {img_path}"})


@router.post("/predictBatch", response_model=str, summary="Cartoon GAN Prediction on Batch of Images", status_code=status.HTTP_202_ACCEPTED, responses={404: {"model": Message}})
def imagePredictionBatch(imageDirectory: str = Query(..., example=r"D:\Projects\cartoon_gan\test\input_images", max_length=100),
                    styleType: dropdownChoices = Form(dropdownChoices.hosoda),
                    maximumSize: Optional[int] = Query(None),
                    directoryToSaveTo: Optional[str] = Query(None, max_length=100),
                    useGPU: Optional[bool] = Query(True)):

    model_dict ={}
    if styleType == "All":
        style_list = ["Hosoda", "Hayao", "Shinkai", "Paprika"]
    else:
        style_list = [styleType]

    for style in style_list:
            model = load_model(style, useGPU)
            model_dict[style] = model

    if os.path.isdir(imageDirectory):
        imageLists = os.listdir(imageDirectory)
        img_num = len(imageLists)
    else:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The directory path {imageDirectory} does not exist"})
    output_dir_final = output_dir
    if directoryToSaveTo:
        output_dir_final = os.path.join(output_dir, directoryToSaveTo)
    if not os.path.isdir(output_dir_final):
        # os.mkdir(output_dir_final)
        os.makedirs(output_dir_final)
    t0 = time.time()
    len_style = len(style_list)
    for style_count, style in enumerate(model_dict.keys()):
        print(f"Processing for style {style_count+1}/{len_style} : {style}")
        model = model_dict[style]
        for count, imageName in enumerate(imageLists):
            img = os.path.join(imageDirectory, imageName)
            if os.path.isfile(img):
                ext = imageName.split(".")[-1]
                output_img, maxS = transform(model, style, img, maximumSize, useGPU)
                if maximumSize:
                    maxS = maximumSize
                img_path = os.path.join(output_dir_final ,".".join(imageName.split(".")[:-1]) + '_' + style.lower() + "_" + str(maxS) + '.' + ext)
                output_img.save(img_path)
                print(f"inference  for image {count+1}/{img_num} took {time.time() - t0} s")
                t0 = time.time()
            else:
                print(f"{count+1}/{img_num} is not an image")
    return f"All the cartoonized images can be found at - {output_dir_final}"


@router.post("/predictOnVideo", summary="Cartoon GAN Prediction on Single Video", status_code=status.HTTP_202_ACCEPTED, responses={404: {"model": Message}})
def imagePredictionVideo(videoFile: UploadFile,
                    videoDirectory: Optional[str] = Query(None, max_length=100),
                    styleType: dropdownChoices = Form(dropdownChoices.hosoda),
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
    temp_dir = vid_input(videoNameOnly, extVid, fr'{videoDirectory}', fps)

    model_dict ={}
    if styleType == "All":
        style_list = ["Hosoda", "Hayao", "Shinkai", "Paprika"]
    else:
        style_list = [styleType]
    
    for style in style_list:
        if not os.path.isdir(f"{videoDirectory}/{temp_dir}_out_{style}"):
            check_output(f"cd {videoDirectory} && mkdir {temp_dir}_out_{style}", shell=True)
    imageDirectory = os.path.join(videoDirectory, temp_dir)

    for style in style_list:
            model = load_model(style, useGPU)
            model_dict[style] = model

    if os.path.isdir(imageDirectory):
        imageLists = os.listdir(imageDirectory)
        img_num = len(imageLists)
    else:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The directory path {imageDirectory} does not exist"})
    output_dir_final = output_video_dir
    if directoryToSaveTo:
        output_dir_final = os.path.join(output_video_dir, directoryToSaveTo)
    if not os.path.isdir(output_dir_final):
        # os.mkdir(output_dir_final)
        os.makedirs(output_dir_final)
    output_path = []
    t0 = time.time()
    len_style = len(style_list)
    for style_count, style in enumerate(model_dict.keys()):
        print(f"Processing for style {style_count+1}/{len_style} : {style}")
        model = model_dict[style]
        for count, imageName in enumerate(imageLists):
            img = os.path.join(imageDirectory, imageName)
            if os.path.isfile(img):
                output_img, maxS = transform(model, style, img, maximumSize, useGPU)
                if not maximumSize:
                    maximumSize = int(maxS)
                img_path = os.path.join(videoDirectory , f"{temp_dir}_out_{style}", imageName)
                output_img.save(img_path)
                print(f"inference  for image {count+1}/{img_num} took {time.time() - t0} s")
                t0 = time.time()
            else:
                print(f"{count+1}/{img_num} is not an image")
        vid_output(videoNameOnly, extVid, fr'{videoDirectory}', fr'{output_dir_final}', style.lower(), maximumSize, fps)
        audio_add(vid, os.path.join(output_dir_final, f"{videoNameOnly}_{style.lower()}_{maximumSize}.{extVid}"))
        output_path.append(os.path.join(output_dir_final, f"{videoNameOnly}_{style.lower()}_{maximumSize}.{extVid}"))
    remove_dirs(fr'{videoDirectory}', style_list)
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


@router.post("/predictOnMultipleVideos", summary="Cartoon GAN Prediction on Multiple Videos", status_code=status.HTTP_202_ACCEPTED, responses={404: {"model": Message}})
def imagePredictionMultipleVideos(videoDirectory: Optional[str] = Query(None, max_length=100),
                    styleType: dropdownChoices = Form(dropdownChoices.hosoda),
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

    model_dict ={}
    if styleType == "All":
        style_list = ["Hosoda", "Hayao", "Shinkai", "Paprika"]
    else:
        style_list = [styleType]
    
    for style in style_list:
        model = load_model(style, useGPU)
        model_dict[style] = model

    output_dir_final = output_video_dir
    if directoryToSaveTo:
        output_dir_final = os.path.join(output_video_dir, directoryToSaveTo)
    if not os.path.isdir(output_dir_final):
        # os.mkdir(output_dir_final)
        os.makedirs(output_dir_final)

    for vidCount, videoName in enumerate(videoList):
        print(f"Processing for video {vidCount+1}/{len_videoList} : {videoName}")
        vid = os.path.join(videoDirectory, videoName)
        if not os.path.isfile(vid):
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The video file path {vid} does not exist"})
        extVid = videoName.split(".")[-1]
        videoNameOnly = ".".join(videoName.split(".")[:-1])
        temp_dir = vid_input(videoNameOnly, extVid, fr'{videoDirectory}', fps)
        for style in style_list:
            if not os.path.isdir(f"{videoDirectory}/{temp_dir}_out_{style}"):
                check_output(f"cd {videoDirectory} && mkdir {temp_dir}_out_{style}", shell=True)
        imageDirectory = os.path.join(videoDirectory, temp_dir)
        if os.path.isdir(imageDirectory):
            imageLists = os.listdir(imageDirectory)
            img_num = len(imageLists)
        else:
            return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": f"The directory path {imageDirectory} does not exist"})
        t0 = time.time()
        len_style = len(style_list)
        for style_count, style in enumerate(model_dict.keys()):
            print(f"Processing for style {style_count+1}/{len_style} : {style}")
            model = model_dict[style]
            for count, imageName in enumerate(imageLists):
                img = os.path.join(imageDirectory, imageName)
                if os.path.isfile(img):
                    output_img, maxS = transform(model, style, img, maximumSize, useGPU)
                    if not maximumSize:
                        maximumSize = int(maxS)
                    img_path = os.path.join(videoDirectory , f"{temp_dir}_out_{style}", imageName)
                    output_img.save(img_path)
                    print(f"inference  for image {count+1}/{img_num} of video {vidCount+1}/{len_videoList} ({videoName}) took {time.time() - t0} s")
                    t0 = time.time()
                else:
                    print(f"{count+1}/{img_num} is not an image")
            vid_output(videoNameOnly, extVid, fr'{videoDirectory}', fr'{output_dir_final}', style.lower(), maximumSize, fps)
            audio_add(vid, os.path.join(output_dir_final, f"{videoNameOnly}_{style.lower()}_{maximumSize}.{extVid}"))
            output_path.append(os.path.join(output_dir_final, f"{videoNameOnly}_{style.lower()}_{maximumSize}.{extVid}"))
        remove_dirs(fr'{videoDirectory}', style_list)
    if downloadZip:
        return zipfiles(output_path)
    else:
        return f"The cartoonized videos can be found at - {output_dir_final}"            

