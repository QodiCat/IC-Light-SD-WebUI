import os
import requests
import shutil


def check_model_structure():
    structure = ''
    model_dir = './extensions/IC-Light-SD-WebUI/models'
    if not os.path.exists(model_dir+'/iclight_sd15_fbc.safetensors'):
        structure += 'iclight_sd15_fbc.safetensors not found\n'
    if not os.path.exists(model_dir+'/iclight_sd15_fbc.safetensors'):
        structure += 'iclight_sd15_fbc.safetensors not found\n'

    if not os.path.exists(model_dir+'/rmbg/config.json'):
        structure += 'rmbg/config.json not found\n'
    if not os.path.exists(model_dir+'/rmbg/model.pth'):
        structure += 'rmbg/model.pth not found\n'
    if not os.path.exists(model_dir+'/rmbg/model.safetensors'):
        structure += 'rmbg/model.safetensors not found\n'
    if not os.path.exists(model_dir+'/rmbg/pytorch_model.bin'):
        structure += 'rmbg/pytorch_model.bin not found\n'

    if not os.path.exists(model_dir+'/text_encoder/config.json'):
        structure += 'text_encoder/config.json not found\n'
    if not os.path.exists(model_dir+'/text_encoder/model.safetensors'):
        structure += 'text_encoder/model.safetensors not found\n'
    if not os.path.exists(model_dir+'/text_encoder/pytorch_model.bin'):
        structure += 'text_encoder/pytorch_model.bin not found\n'

    if not os.path.exists(model_dir+'/tokenizer/special_tokens_map.json'):
        structure += 'tokenizer/special_tokens_map.json not found\n'
    if not os.path.exists(model_dir+'/tokenizer/tokenizer_config.json'):
        structure += 'tokenizer/tokenizer_config.json not found\n'
    if not os.path.exists(model_dir+'/tokenizer/vocab.json'):
        structure += 'tokenizer/vocab.json not found\n'
    if not os.path.exists(model_dir+'/tokenizer/merges.txt'):
        structure += 'tokenizer/merges.txt not found\n'

    if not os.path.exists(model_dir+'/unet/config.json'):
        structure += 'unet/config.json not found\n'
    if not os.path.exists(model_dir+'/unet/diffusion_pytorch_model.bin'):
        structure += 'unet/diffusion_pytorch_model.bin not found\n'
    if not os.path.exists(model_dir+'/unet/diffusion_pytorch_model.safetensors'):
        structure += 'unet/diffusion_pytorch_model.safetensors not found\n'

    if not os.path.exists(model_dir+'/vae/config.json'):
        structure += 'vae/config.json not found\n'
    if not os.path.exists(model_dir+'/vae/diffusion_pytorch_model.bin'):
        structure += 'vae/diffusion_pytorch_model.bin not found\n'
    if not os.path.exists(model_dir+'/vae/diffusion_pytorch_model.safetensors'):
        structure += 'vae/diffusion_pytorch_model.safetensors not found\n'
    if structure == '':
        structure = 'All files found'
    return structure

def dl(path, url):
    with open(path, 'wb') as out_file:
        out_file.write(requests.get(url, stream=True).text.encode('utf-8'))


def download_model():
    model_dir = './extensions/IC-Light-SD-WebUI/models'
    if not os.path.exists(model_dir+'/iclight_sd15_fbc.safetensors'):
        dl("./extensions/IC-Light-SD-WebUI/models/iclight_sd15_fc.safetensors","https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors")
    if not os.path.exists(model_dir+'/iclight_sd15_fbc.safetensors'):
        dl("./extensions/IC-Light-SD-WebUI/models/iclight_sd15_fbc.safetensors","https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors")

    if not os.path.exists(model_dir+'/rmbg/config.json'):
        dl("./extensions/IC-Light-SD-WebUI/models/rmbg/config.json","https://huggingface.co/briaai/RMBG-1.4/resolve/main/config.json")
    if not os.path.exists(model_dir+'/rmbg/model.pth'):
        dl("./extensions/IC-Light-SD-WebUI/models/rmbg/model.pth","https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth")
    if not os.path.exists(model_dir+'/rmbg/model.safetensors'):
        dl("./extensions/IC-Light-SD-WebUI/models/rmbg/model.safetensors","https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.safetensors")
    if not os.path.exists(model_dir+'/rmbg/pytorch_model.bin'):
        dl("./extensions/IC-Light-SD-WebUI/models/rmbg/pytorch_model.bin","https://huggingface.co/briaai/RMBG-1.4/resolve/main/pytorch_model.bin")

    if not os.path.exists(model_dir+'/text_encoder/config.json'):
        dl("./extensions/IC-Light-SD-WebUI/models/text_encoder/config.json","https://huggingface.co/stablediffusionapi/realistic-vision-v51/resolve/main/text_encoder/config.json")
    if not os.path.exists(model_dir+'/text_encoder/model.safetensors'):
        dl("./extensions/IC-Light-SD-WebUI/models/text_encoder/model.safetensors","https://huggingface.co/stablediffusionapi/realistic-vision-v51/resolve/main/text_encoder/model.safetensors")
    if not os.path.exists(model_dir+'/text_encoder/pytorch_model.bin'):
        dl("./extensions/IC-Light-SD-WebUI/models/text_encoder/pytorch_model.bin","https://huggingface.co/stablediffusionapi/realistic-vision-v51/resolve/main/text_encoder/pytorch_model.bin")

    if not os.path.exists(model_dir+'/tokenizer/special_tokens_map.json'):
        dl("./extensions/IC-Light-SD-WebUI/models/tokenizer/special_tokens_map.json","https://huggingface.co/stablediffusionapi/realistic-vision-v51/resolve/main/tokenizer/special_tokens_map.json")
    if not os.path.exists(model_dir+'/tokenizer/tokenizer_config.json'):
        dl("./extensions/IC-Light-SD-WebUI/models/tokenizer/tokenizer_config.json","https://huggingface.co/stablediffusionapi/realistic-vision-v51/resolve/main/tokenizer/tokenizer_config.json")
    if not os.path.exists(model_dir+'/tokenizer/vocab.json'):       
        dl("./extensions/IC-Light-SD-WebUI/models/tokenizer/vocab.json","https://huggingface.co/stablediffusionapi/realistic-vision-v51/resolve/main/tokenizer/vocab.json")
    if not os.path.exists(model_dir+'/tokenizer/merges.txt'):
        dl("./extensions/IC-Light-SD-WebUI/models/tokenizer/merges.txt","https://huggingface.co/stablediffusionapi/realistic-vision-v51/resolve/main/tokenizer/merges.txt")

    if not os.path.exists(model_dir+'/unet/config.json'):
        dl("./extensions/IC-Light-SD-WebUI/models/unet/config.json","https://huggingface.co/stablediffusionapi/realistic-vision-v51/resolve/main/unet/config.json")
    if not os.path.exists(model_dir+'/unet/diffusion_pytorch_model.bin'):
        dl("./extensions/IC-Light-SD-WebUI/models/unet/diffusion_pytorch_model.bin","https://huggingface.co/stablediffusionapi/realistic-vision-v51/resolve/main/unet/diffusion_pytorch_model.bin")
    if not os.path.exists(model_dir+'/unet/diffusion_pytorch_model.safetensors'):
        dl("./extensions/IC-Light-SD-WebUI/models/unet/diffusion_pytorch_model.safetensors","https://huggingface.co/stablediffusionapi/realistic-vision-v51/resolve/main/unet/diffusion_pytorch_model.safetensors")

    if not os.path.exists(model_dir+'/vae/config.json'):
        dl("./extensions/IC-Light-SD-WebUI/models/vae/config.json","https://huggingface.co/stablediffusionapi/realistic-vision-v51/resolve/main/vae/config.json")
    if not os.path.exists(model_dir+'/vae/diffusion_pytorch_model.bin'):
        dl("./extensions/IC-Light-SD-WebUI/models/vae/diffusion_pytorch_model.bin","https://huggingface.co/stablediffusionapi/realistic-vision-v51/resolve/main/vae/diffusion_pytorch_model.bin")
    if not os.path.exists(model_dir+'/vae/diffusion_pytorch_model.safetensors'):
        dl("./extensions/IC-Light-SD-WebUI/models/vae/diffusion_pytorch_model.safetensors","https://huggingface.co/stablediffusionapi/realistic-vision-v51/resolve/main/vae/diffusion_pytorch_model.safetensors")


def download_mirror_model():
    model_dir = './extensions/IC-Light-SD-WebUI/models'
    if not os.path.exists(model_dir+'/iclight_sd15_fbc.safetensors'):
        dl("./extensions/IC-Light-SD-WebUI/models/iclight_sd15_fc.safetensors","https://hf-mirror.com/spaces/lllyasviel/IC-Lightresolve/main/iclight_sd15_fc.safetensors")
    if not os.path.exists(model_dir+'/iclight_sd15_fbc.safetensors'):
        dl("./extensions/IC-Light-SD-WebUI/models/iclight_sd15_fbc.safetensors","https://hf-mirror.com/spaces/lllyasviel/IC-Lightresolve/main/iclight_sd15_fbc.safetensors")

    if not os.path.exists(model_dir+'/rmbg/config.json'):
        dl("./extensions/IC-Light-SD-WebUI/models/rmbg/config.json","https://hf-mirror.com/briaai/RMBG-1.4/resolve/main/config.json")
    if not os.path.exists(model_dir+'/rmbg/model.pth'):
        dl("./extensions/IC-Light-SD-WebUI/models/rmbg/model.pth","https://hf-mirror.com/briaai/RMBG-1.4/resolve/main/model.pth")
    if not os.path.exists(model_dir+'/rmbg/model.safetensors'):
        dl("./extensions/IC-Light-SD-WebUI/models/rmbg/model.safetensors","https://hf-mirror.com/briaai/RMBG-1.4/resolve/main/model.safetensors")
    if not os.path.exists(model_dir+'/rmbg/pytorch_model.bin'):
        dl("./extensions/IC-Light-SD-WebUI/models/rmbg/pytorch_model.bin","https://hf-mirror.com/briaai/RMBG-1.4/resolve/main/pytorch_model.bin")

    if not os.path.exists(model_dir+'/text_encoder/config.json'):
        dl("./extensions/IC-Light-SD-WebUI/models/text_encoder/config.json","https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/resolve/main/text_encoder/config.json")
    if not os.path.exists(model_dir+'/text_encoder/model.safetensors'):
        dl("./extensions/IC-Light-SD-WebUI/models/text_encoder/model.safetensors","https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/resolve/main/text_encoder/model.safetensors")
    if not os.path.exists(model_dir+'/text_encoder/pytorch_model.bin'):
        dl("./extensions/IC-Light-SD-WebUI/models/text_encoder/pytorch_model.bin","https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/resolve/main/text_encoder/pytorch_model.bin")

    if not os.path.exists(model_dir+'/tokenizer/special_tokens_map.json'):
        dl("./extensions/IC-Light-SD-WebUI/models/tokenizer/special_tokens_map.json","https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/resolve/main/tokenizer/special_tokens_map.json")
    if not os.path.exists(model_dir+'/tokenizer/tokenizer_config.json'):
        dl("./extensions/IC-Light-SD-WebUI/models/tokenizer/tokenizer_config.json","https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/resolve/main/tokenizer/tokenizer_config.json")
    if not os.path.exists(model_dir+'/tokenizer/vocab.json'):
        dl("./extensions/IC-Light-SD-WebUI/models/tokenizer/vocab.json","https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/resolve/main/tokenizer/vocab.json")
    if not os.path.exists(model_dir+'/tokenizer/merges.txt'):
        dl("./extensions/IC-Light-SD-WebUI/models/tokenizer/merges.txt","https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/resolve/main/tokenizer/merges.txt")
    if not os.path.exists(model_dir+'/unet/config.json'):
        dl("./extensions/IC-Light-SD-WebUI/models/unet/config.json","https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/resolve/main/unet/config.json")
    if not os.path.exists(model_dir+'/unet/diffusion_pytorch_model.bin'):
        dl("./extensions/IC-Light-SD-WebUI/models/unet/diffusion_pytorch_model.bin","https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/resolve/main/unet/diffusion_pytorch_model.bin")
    if not os.path.exists(model_dir+'/unet/diffusion_pytorch_model.safetensors'):
        dl("./extensions/IC-Light-SD-WebUI/models/unet/diffusion_pytorch_model.safetensors","https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/resolve/main/unet/diffusion_pytorch_model.safetensors")

    if not os.path.exists(model_dir+'/vae/config.json'):
        dl("./extensions/IC-Light-SD-WebUI/models/vae/config.json","https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/resolve/main/vae/config.json")
    if not os.path.exists(model_dir+'/vae/diffusion_pytorch_model.bin'):
        dl("./extensions/IC-Light-SD-WebUI/models/vae/diffusion_pytorch_model.bin","https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/resolve/main/vae/diffusion_pytorch_model.bin")
    if not os.path.exists(model_dir+'/vae/diffusion_pytorch_model.safetensors'):
        dl("./extensions/IC-Light-SD-WebUI/models/vae/diffusion_pytorch_model.safetensors","https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/resolve/main/vae/diffusion_pytorch_model.safetensors")

