import os
import subprocess
from subprocess import check_output
from moviepy.editor import VideoFileClip, AudioFileClip

def audio_add(video_extract, video_wo_audio):
    audio_file = "audio.wav"

    video_wo_audio_name = ".".join(video_wo_audio.split(".")[:-1])
    video_wo_audio_ext = video_wo_audio.split(".")[-1]


    inp= f"ffprobe -i {video_extract} -show_streams -select_streams a -loglevel error"
    ifAudio = check_output(inp, shell=True)

    if len(ifAudio) != 0:
        try:
            command = f"ffmpeg -i {video_extract} -ab 160k -ac 2 -ar 44100 -vn {audio_file}"

            subprocess.call(command, shell=True)
            
            clip = VideoFileClip(video_wo_audio)
            
            audioclip = AudioFileClip(audio_file)
            
            videoclip = clip.set_audio(audioclip)

            final_video = f"{video_wo_audio_name}_temp.{video_wo_audio_ext}"

            videoclip.write_videofile(final_video)

            os.remove(audio_file)
            os.remove(video_wo_audio)
            os.rename(final_video, video_wo_audio)
            return True
        except:
            return False
    return False

