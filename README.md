# IC-Light-SD-WebUI

 [IC-Light](https://github.com/lllyasviel/IC-Light) is a project to manipulate the illumination of images.

We know that SD can generate beautiful character and landscape images, so there is this extension integrated into SD-WebUI.

## Download models

To use it in SD-WebUI,you need download following contents after you put this project under extensions :

models:

* [iclight_sd15_fc.safetensors](https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fc.safetensors?download=true)

* [iclight_sd15_fbc.safetensors](https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors?download=true)

* [unet](https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/tree/main/unet)
* [vae](https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/tree/main/vae)
* [tokenizer](https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/tree/main/tokenizer)
* [text_encoder](https://hf-mirror.com/stablediffusionapi/realistic-vision-v51/tree/main/text_encoder)

* [rmbg](https://huggingface.co/briaai/RMBG-1.4/tree/main)
  * config.json
  * model.pth
  * model.safetensors
  * pytorch_model.bin

## How to use

IC-Light has two workflows. The FC workflow directly extracts characters from the image and generates a new photo based on prompt words.



First, we use SD to generate a good-looking picture. Then, we open the IC-Light tab, load the model and the just-generated image in order, add a few prompt words, and then we can get a picture that feels different from before.

![1](imgs/1.png)

![2](imgs/2.png)

Unlike the FC workflow, the FBC workflow allows us to import a specified background image and does not modify it.

![3](imgs/3.png)





## 国内用户，可以直接从百度网盘下载整合包

链接： https://pan.baidu.com/s/1kb3rBi3MIG2Dl5b-DSiCyg?pwd=1111 

