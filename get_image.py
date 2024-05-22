import os
import glob
from pathlib import Path
from PIL import Image
from modules import shared, paths, script_callbacks
import time

def get_img_from_txt2img():
    talker_path = str(paths.script_path) + "/outputs"
    dateNow = time.strftime("%Y-%m-%d", time.localtime())
    if os.path.exists(talker_path + "/txt2img-images/" +dateNow):
        dir = talker_path + "/txt2img-images/" + dateNow
    else:
        dir = talker_path + "/txt2img-images"
    files = glob.glob(os.path.join(dir, '*'))
    latest_file = max(files, key=os.path.getmtime)
    return Image.open(latest_file)

def get_img_from_img2img(x):
    talker_path = str(paths.script_path) + "/outputs"
    dateNow = time.strftime("%Y-%m-%d", time.localtime())
    if os.path.exists(talker_path + "/img2img-images/" +dateNow):
        dir = talker_path + "/img2img-images/" + dateNow
    else:
        dir = talker_path + "/img2img-images"
    files = glob.glob(os.path.join(dir, '*'))
    latest_file = max(files, key=os.path.getmtime)
    return Image.open(latest_file)