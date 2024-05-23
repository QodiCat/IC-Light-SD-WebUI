import os
import glob
from pathlib import Path
from PIL import Image
from modules import shared, paths, script_callbacks


def get_img_from_txt2img():
    talker_path = str(paths.script_path) + "/outputs"
    dir = talker_path + "/txt2img-images"
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(dir)) for f in fn]
    latest_file = max(files, key=os.path.getmtime)
    return Image.open(latest_file)

def get_img_from_img2img(x):
    talker_path = str(paths.script_path) + "/outputs"
    dir = talker_path + "/img2img-images"
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(dir)) for f in fn]
    latest_file = max(files, key=os.path.getmtime)
    return Image.open(latest_file)