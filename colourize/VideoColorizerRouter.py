from fastapi import FastAPI, Query, APIRouter, status
from typing import Optional, List, final
import shutil
import os

from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import *
import warnings


router = APIRouter(tags = ["Video Colourization"])

@router.post("/colourizeVideo", summary="Colourize Video", status_code=status.HTTP_202_ACCEPTED)
def colourizeVideoFunction(videoDirectory: str = Query(..., max_length=100),
                    videoName: str = Query(..., max_length=100),
                    extension: str = Query(..., max_length=20),
                    gpuID: Optional[int] = Query(0),
                    directoryToSaveTo: Optional[str] = Query(None, max_length=100)):

    root_video_dir = "video"

    shutil.copy(f"{videoDirectory}/{videoName}.{extension}", f"{root_video_dir}/source/{videoName}.{extension}")
    
    device.set(device=DeviceId.GPU0)

    # if gpuID == 0:
    #     device.set(device=DeviceId.GPU0)
    # elif gpuID == 1:
    #     device.set(device=DeviceId.GPU1)
    # elif gpuID == 2:
    #     device.set(device=DeviceId.GPU2)
    # elif gpuID == 3:
    #     device.set(device=DeviceId.GPU3)
    # elif gpuID == 4:
    #     device.set(device=DeviceId.GPU4)
    # elif gpuID == 5:
    #     device.set(device=DeviceId.GPU5)
    # elif gpuID == 6:
    #     device.set(device=DeviceId.GPU6)
    # elif gpuID == 7:
    #     device.set(device=DeviceId.GPU7)
    # else:
    #     device.set(device=DeviceId.CPU)
    
    try:
        plt.style.use('dark_background')

        warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

        colorizer = get_video_colorizer()

        render_factor=21
        file_name_ext = videoName + '.' + extension

        colorizer.colorize_from_file_name(file_name_ext, render_factor=render_factor)

        shutil.rmtree(f"{root_video_dir}/bwframes/{videoName}")
        shutil.rmtree(f"{root_video_dir}/colorframes/{videoName}")
        for file in os.listdir(f"{root_video_dir}/source"):
            os.remove(f"{root_video_dir}/source/{file}")
        os.remove(f"{root_video_dir}/result/{videoName}_no_audio.{extension}")
        shutil.move(f"{root_video_dir}/result/{videoName}.{extension}", directoryToSaveTo)
        
        return True
    except:
        return False
    
    




