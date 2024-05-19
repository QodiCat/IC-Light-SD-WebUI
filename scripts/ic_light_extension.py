import os, sys
from pathlib import Path
import gradio as gr
from modules import scripts, script_callbacks
from modules.shared import opts
import launch
from modules import paths
from enum import Enum
import torch
from gradio_demo import iclight_demo





def install():
    kv = {
        "diffusers": "diffusers==0.27.2",
        "opencv-python": "opencv-python",
        "safetensors": "safetensors",
        "pillow": "pillow==10.2.0",
        "einops": "einops",
        "peft": "peft",
        "protobuf": "protobuf==3.20",
    }

        

    for k,v in kv.items():

        if not launch.is_installed(k):
                print(k, launch.is_installed(k))
                launch.run_pip("install "+ v, "requirements for IC-Light-SD-WebUI")

        if k == 'diffusers':
            import pkg_resources
            version = pkg_resources.get_distribution('diffusers').version
            if version != '0.27.2':
                launch.run_pip("uninstall -y diffusers", "uninstalling diffusers due to version mismatch")
                launch.run_pip("install "+"diffusers==0.27.2")
        
    

def on_ui_tab():
    install()

    ic_light_view = iclight_demo()

    return [(ic_light_view, "IC-Light-SD-WebUI","IC_LIGHT_extension")]


script_callbacks.on_ui_tabs(on_ui_tab)