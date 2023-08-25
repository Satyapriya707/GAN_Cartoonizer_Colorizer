import os
import shutil
from subprocess import check_output
from uuid import uuid4

dirName = str(uuid4())

def vid_input(name, ext, input_dir, fps, d=dirName):
    if not os.path.isdir(f"{input_dir}/{d}"):
        check_output(f"cd {input_dir} && mkdir {d}", shell=True)
    full_path = os.path.join(input_dir, d)
    inp = f'ffmpeg -i "{input_dir}/{name}.{ext}" -r {fps} {full_path}/${name}%03d.bmp'
    check_output(inp, shell=True)
    return d

def vid_output(name, ext, input_dir, output_dir, style, max_size, fps, d=dirName):    
    d_new = f"{d}_out_{style}"
    full_path = os.path.join(input_dir, d_new)
    out_name = f"{output_dir}/{name}_{style}_{max_size}"
    if ext == "gif":
        out = f'ffmpeg -r {fps} -i "{full_path}/${name}%3d.bmp" {out_name}.{ext}'
    else:
        out = f'ffmpeg -r {fps} -i "{full_path}/${name}%3d.bmp" -c:v libx264 -pix_fmt yuv420p {out_name}.{ext}'        
    check_output(out, shell=True)
    return 0

def vid_output_wb(name, ext, input_dir, output_dir, style, max_size, fps, d=dirName):    
    d_new = f"{d}_out"
    full_path = os.path.join(input_dir, d_new)
    out_name = f"{output_dir}/{name}_{style}_{max_size}"
    if ext == "gif":
        out = f'ffmpeg -r {fps} -i "{full_path}/${name}%3d_out.bmp" {out_name}.{ext}'
    else:
        out = f'ffmpeg -r {fps} -i "{full_path}/${name}%3d_out.bmp" -c:v libx264 -pix_fmt yuv420p {out_name}.{ext}'        
    check_output(out, shell=True)
    return 0

def vid_output_wb_tf(name, ext, input_dir, output_dir, max_size, fps, d=dirName):    
    d_new = f"{d}_out"
    full_path = os.path.join(input_dir, d_new)
    out_name = f"{output_dir}/{name}_{max_size}"
    if ext == "gif":
        out = f'ffmpeg -r {fps} -i "{full_path}/${name}%3d_out.bmp" {out_name}.{ext}'
    else:
        out = f'ffmpeg -r {fps} -i "{full_path}/${name}%3d_out.bmp" -c:v libx264 -pix_fmt yuv420p {out_name}.{ext}'        
    check_output(out, shell=True)
    return 0

def remove_dirs(input_dir, style_list, d=dirName):
    shutil.rmtree(os.path.join(input_dir,d))
    for style in style_list:
        d_new = f"{d}_out_{style}"
        shutil.rmtree(os.path.join(input_dir,d_new))


