import gradio as gr
import os
import numpy as np
from pydub import AudioSegment
import hashlib
from dice_talk import DICE_Talk
import shutil

pipe = DICE_Talk(0)


def get_md5(content):
    md5hash = hashlib.md5(content)
    md5 = md5hash.hexdigest()
    return md5

def get_video_res(img_path, audio_path, emotion_path, res_video_path, ref_scale=None, emo_scale=None, crop=False):

    expand_ratio = 0.5
    min_resolution = 512
    inference_steps = 25

    face_info = pipe.preprocess(img_path, expand_ratio=expand_ratio)
    print(face_info)
    if face_info['face_num'] > 0:
        if crop:
            crop_image_path = img_path + '.crop.png'
            pipe.crop_image(img_path, crop_image_path, face_info['crop_bbox'])
            img_path = crop_image_path
        os.makedirs(os.path.dirname(res_video_path), exist_ok=True)
        pipe.process(img_path, audio_path, emotion_path, res_video_path, min_resolution=min_resolution, inference_steps=inference_steps, ref_scale=ref_scale, emo_scale=emo_scale)
    else:
        return -1
tmp_path = './tmp_path/'
res_path = './res_path/'
os.makedirs(tmp_path,exist_ok=1)
os.makedirs(res_path,exist_ok=1)

def process_dice(image,audio, emotion, s0, s1, crop=False):
    img_md5= get_md5(np.array(image))
    audio_md5 = get_md5(audio[1])

    print(img_md5,audio_md5)
    sampling_rate, arr = audio[:2]
    if len(arr.shape)==1:
        arr = arr[:,None]
    audio = AudioSegment(
        arr.tobytes(),
        frame_rate=sampling_rate,
        sample_width=arr.dtype.itemsize,
        channels=arr.shape[1]
    )
    audio = audio.set_frame_rate(sampling_rate)
    image_path = os.path.abspath(tmp_path+'{0}.png'.format(img_md5))
    audio_path = os.path.abspath(tmp_path+'{0}.wav'.format(audio_md5))
    emotion_path = os.path.abspath(tmp_path+'{0}.npy'.format(emotion))
    if not os.path.exists(image_path):
        image.save(image_path)
    if not os.path.exists(audio_path):
        audio.export(audio_path, format="wav")
    if not os.path.exists(emotion_path):
        shutil.copy(f'examples/emo/{emotion}.npy', emotion_path)
    res_video_path = os.path.abspath(res_path+f'{img_md5}_{audio_md5}_{emotion}_{s0}_{s1}_{int(crop)}.mp4')
    if os.path.exists(res_video_path):
        return res_video_path
    else:
        get_video_res(image_path, audio_path, emotion_path, res_video_path, s0, s1, crop=crop)
    return res_video_path
    
inputs = [
    gr.Image(type='pil',label="Upload Image"),
    gr.Audio(label="Upload Audio"),
    gr.Dropdown(
        choices=['contempt', 'sad', 'happy', 'surprised', 'angry', 'disgusted', 'fear', 'neutral'],
        label="Choose Emotion"
    ),
    gr.Slider(0.0, 10.0, value=3.0, step=0.1, label="Reference", info="Increase/decrease to obtain stronger/weaker identity preservation"),
    gr.Slider(0.0, 10.0, value=6.0, step=0.1, label="Emotion", info="Increase/decrease to obtain stronger/weaker emotions"),
    gr.Checkbox(label="Crop image", value=False)
]
outputs = gr.Video(label="output.mp4")


html_description = """
<div style="display: flex; justify-content: center; align-items: center;">
  <a href="https://github.com/toto222/DICE-Talk" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <a href="https://arxiv.org/abs/2504.18087" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2504.18087-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://toto222.github.io/DICE-Talk/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
  <a href="https://github.com/toto222/DICE-Talk/blob/main/LICENSE" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
  </a>
</div>

The demo can only be used for <b>Non-commercial Use</b>.
<br>If you like our work, please star <a href='https://github.com/toto222/DICE-Talk' style="margin: 0 2px;">DICE-Talk</a>.
"""

def get_example():
    return [
        ["examples/img/nazha.png", "examples/wav/female-zh.wav", "happy", 3.0, 6.0],
        ["examples/img/pyy.jpg", "examples/wav/male-zh.wav", "neutral", 4.0, 7.5],
        ["examples/img/female.png", "examples/wav/female.wav","happy", 3.0, 6.0],
        ["examples/img/hg.jpeg", "examples/wav/male.wav", "surprised", 3.0, 6.0],
        
    ]

with gr.Blocks(title="DICE-Talk") as demo:
    gr.Interface(fn=process_dice, inputs=inputs, outputs=outputs, title="Disentangle Identity, Cooperate Emotion: Correlation-Aware\
        Emotional Talking Portrait Generation", description=html_description)
    gr.Examples(
        examples=get_example(),
        fn=process_dice,
        inputs=inputs,
        outputs=outputs,
        cache_examples=False,)

demo.queue()
demo.launch(server_name='0.0.0.0', server_port=8081, share=False)