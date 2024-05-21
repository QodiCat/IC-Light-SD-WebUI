import gradio as gr
from enum import Enum
from process_fc import process_relight,change_state1
from process_fbc import change_state2,process_relight2
from get_image import get_img_from_txt2img, get_img_from_img2img
from check_model import check_model_structure
from check_model import download_model
from check_model import download_mirror_model

class BGSource_1(Enum):
    NONE = "None"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"

class BGSource_2(Enum):
    UPLOAD = "Use Background Image"
    UPLOAD_FLIP = "Use Flipped Background Image"
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    GREY = "Ambient"

def iclight_demo(checkpoint_path="checkpoints",config_path="src/config",warpfn=None):
    
    with gr.Blocks() as ic_light_view:
        with gr.Tab(label="IC-light-FC"):
            with gr.Row():
                with gr.Column():
                    if_used=gr.Checkbox(label="Use FC", show_label=True)
                    if_used.change(change_state1,[if_used],[])
                    with gr.Row():
                        input_fg = gr.Image(source='upload', type="numpy", label="Image", height=480)
                        output_bg = gr.Image(type="numpy", label="Preprocessed Foreground", height=480)
                    with gr.Row():
                        upload_button_text=gr.Button(value="from txt_to_img")
                        upload_button_img=gr.Button(value="from img_to_img")

                        upload_button_text.click(fn=get_img_from_txt2img, inputs=[], outputs=[input_fg])
                        upload_button_img.click(fn=get_img_from_img2img, inputs=[], outputs=[input_fg])
                    prompt = gr.Textbox(label="Prompt")
                    bg_source = gr.Radio(choices=[e.value for e in BGSource_1],
                                        value=BGSource_1.NONE.value,
                                        label="Lighting Preference (Initial Latent)", type='value')
                    relight_button = gr.Button(value="Relight")

                    with gr.Group():
                        with gr.Row():
                            num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                            seed = gr.Number(label="Seed", value=12345, precision=0)

                        with gr.Row():
                            image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=512, step=64)
                            image_height = gr.Slider(label="Image Height", minimum=256, maximum=1024, value=640, step=64)

                    with gr.Accordion("Advanced options", open=False):
                        steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                        cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=2, step=0.01)
                        lowres_denoise = gr.Slider(label="Lowres Denoise (for initial latent)", minimum=0.1, maximum=1.0, value=0.9, step=0.01)
                        highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                        highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=1.0, value=0.5, step=0.01)
                        a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                        n_prompt = gr.Textbox(label="Negative Prompt", value='lowres, bad anatomy, bad hands, cropped, worst quality')
                with gr.Column():
                    result_gallery = gr.Gallery(height=832, object_fit='contain', label='Outputs')
                    
                    
        
            ips = [input_fg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, lowres_denoise, bg_source]
            relight_button.click(fn=process_relight, inputs=ips, outputs=[output_bg, result_gallery])
       
        with gr.Tab(label="IC-light-FBC"):
            with gr.Row():
                with gr.Column():
                    if_used2=gr.Checkbox(label="Use FBC", show_label=True)
                    if_used2.change(change_state2,[if_used2],[])
                    with gr.Row():
                        input_fg2 = gr.Image(source='upload', type="numpy", label="Foreground", height=480)
                        input_bg2 = gr.Image(source='upload', type="numpy", label="Background", height=480)
                    with gr.Row():
                        upload_button_text_fg=gr.Button(value="t2i_to_foreground")
                        upload_button_img_fg=gr.Button(value="i2i_to_foreground")
                        upload_button_text_bg=gr.Button(value="t2i_to_background")
                        upload_button_img_bg=gr.Button(value="t2i_to_background")
                        upload_button_text_fg.click(fn=get_img_from_txt2img, inputs=[], outputs=[input_fg2])
                        upload_button_img_fg.click(fn=get_img_from_img2img, inputs=[], outputs=[input_fg2])
                        upload_button_text_bg.click(fn=get_img_from_txt2img, inputs=[], outputs=[input_bg2])
                        upload_button_img_bg.click(fn=get_img_from_img2img, inputs=[], outputs=[input_bg2])
                    prompt = gr.Textbox(label="Prompt")
                    bg2_source = gr.Radio(choices=[e.value for e in BGSource_2],
                                        value=BGSource_2.UPLOAD.value,
                                        label="Background Source", type='value')
                    relight_button2 = gr.Button(value="Relight")
                    with gr.Group():
                        with gr.Row():
                            num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                            seed = gr.Number(label="Seed", value=12345, precision=0)
                        with gr.Row():
                            image_width = gr.Slider(label="Image Width", minimum=256, maximum=1024, value=512, step=64)
                            image_height = gr.Slider(label="Image Height", minimum=256, maximum=1024, value=640, step=64)

                    with gr.Accordion("Advanced options", open=False):
                        steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                        cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=7.0, step=0.01)
                        highres_scale = gr.Slider(label="Highres Scale", minimum=1.0, maximum=3.0, value=1.5, step=0.01)
                        highres_denoise = gr.Slider(label="Highres Denoise", minimum=0.1, maximum=0.9, value=0.5, step=0.01)
                        a_prompt = gr.Textbox(label="Added Prompt", value='best quality')
                        n_prompt = gr.Textbox(label="Negative Prompt",
                                            value='lowres, bad anatomy, bad hands, cropped, worst quality')
                        normal_button = gr.Button(value="Compute Normal (4x Slower)")
                with gr.Column():
                    result_gallery = gr.Gallery(height=832, object_fit='contain', label='Outputs')


        with gr.Tab(label="configuration"):
            with gr.Row():
                with gr.Column():
                    check_button = gr.Button(value="check models")
                    download_button = gr.Button(value="download model from huggingface")
                    download_mir_button = gr.Button(value="download model from hf-mirror")               
                with gr.Column():
                    check_result = gr.Textbox(label="check result")
                    check_button.click(check_model_structure, inputs=None, outputs=check_result)
                    download_button.click(download_model, inputs=None, outputs=None)
                    download_mir_button.click(download_mirror_model, inputs=None, outputs=None)
            
            
            ips2 = [input_fg2, input_bg2, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg2_source]
            relight_button2.click(fn=process_relight2, inputs=ips2, outputs=[result_gallery])
    return ic_light_view