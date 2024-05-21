# IC-Light-SD-WebUI

 [IC-Light](https://github.com/lllyasviel/IC-Light) is a project to manipulate the illumination of images.

We know that SD can generate beautiful character and landscape images, so there is this extension integrated into SD-WebUI.

## Download models

To use it in SD-WebUI,you need download many contents.

Just click "download model from huggingface" and wait.

![4](imgs/4.png)

If all models right,click "check models" and "check result" will be "All files found".

![5](imgs/5.png)



Make sure the folder structure like this:

IC-Light-SD-WebUI
   │  briarmbg.py
   │  check_model.py
   │  get_image.py
   │  gradio_demo.py
   │  process_fbc.py
   │  process_fc.py
   ├─models
   │  │  iclight_sd15_fbc.safetensors
   │  │  iclight_sd15_fc.safetensors
   │  │
   │  ├─rmbg
   │  │      config.json
   │  │      model.pth
   │  │      model.safetensors
   │  │      pytorch_model.bin
   │  │
   │  ├─text_encoder
   │  │      config.json
   │  │      model.safetensors
   │  │      pytorch_model.bin
   │  │
   │  ├─tokenizer
   │  │      merges.txt
   │  │      special_tokens_map.json
   │  │      tokenizer_config.json
   │  │      vocab.json
   │  │
   │  ├─unet
   │  │      config.json
   │  │      diffusion_pytorch_model.bin
   │  │      diffusion_pytorch_model.safetensors
   │  │
   │  └─vae
   │          config.json
   │          diffusion_pytorch_model.bin
   │          diffusion_pytorch_model.safetensors
   │
   └─scripts

​          ic_light_extension.py



## How to use

IC-Light has two workflows. The FC workflow directly extracts characters from the image and generates a new photo based on prompt words.



First, we use SD to generate a good-looking picture. Then, we open the IC-Light tab, load the model and the just-generated image in order, add a few prompt words, and then we can get a picture that feels different from before.

![1](imgs/1.png)

![2](imgs/2.png)

Unlike the FC workflow, the FBC workflow allows us to import a specified background image and does not modify it.

![3](imgs/3.png)

## Notice

* You must click "Use FC"/"Use FBC" first
* "download from hf-mirror"  can't download iclight_sd15_fc.safetensors and iclight_sd15_fbc.safetensors,you should download it from huggingface

## 国内用户，可以直接从百度网盘下载整合包

链接： https://pan.baidu.com/s/1kb3rBi3MIG2Dl5b-DSiCyg?pwd=1111 

将整合包解压放在extensions目录下

