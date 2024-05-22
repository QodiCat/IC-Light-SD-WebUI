import os


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

