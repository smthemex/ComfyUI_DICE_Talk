# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import numpy as np
from omegaconf import OmegaConf
from diffusers import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import WhisperModel, AutoFeatureExtractor
import random
import io
import torchaudio

from .dice_talk import DICE_Talk,preprocess_face,crop_face_image,dice_talk_predata
from .src.dataset.test_preprocess import process_bbox, image_audio_emo_to_tensor
from .src.models.base.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel, add_ip_adapters
from .src.models.audio_adapter.pose_guider import PoseGuider
from .src.utils.face_align.align import AlignImage
from .src.models.emotion_adapter.emo import EmotionModel
from .src.models.audio_adapter.audio_proj import AudioProjModel


from .node_utils import convert_cf2diffuser,gc_clear,tensor2cv,cv2pil,tensor2pil,tensor_upscale
import folder_paths

MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))


device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

# add checkpoints dir
Dice_Talk_weigths_path = os.path.join(folder_paths.models_dir, "dice_talk")
if not os.path.exists(Dice_Talk_weigths_path):
    os.makedirs(Dice_Talk_weigths_path)
folder_paths.add_model_folder_path("dice_talk", Dice_Talk_weigths_path)

# use same model for all nodes
SONIC_weigths_path = os.path.join(folder_paths.models_dir, "sonic")
if os.path.exists(SONIC_weigths_path):
    folder_paths.add_model_folder_path("sonic", SONIC_weigths_path)
else:
    SONIC_weigths_path=""

class Dice_Talk_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "dice_talk_unet": (["none"] + folder_paths.get_filename_list("dice_talk"),),
                "ip_audio_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "ip_emo_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "use_interframe": ("BOOLEAN", {"default": True},),
                "dtype": (["fp16", "fp32", "bf16"],),
            },
        }

    RETURN_TYPES = ("MODEL_DICETALK","DTYPE")
    RETURN_NAMES = ("model","weight_dtype")
    FUNCTION = "loader_main"
    CATEGORY = "Dice_Talk"

    def loader_main(self, model, dice_talk_unet, ip_audio_scale,ip_emo_scale, use_interframe, dtype):

        if dtype == "fp16":
            weight_dtype = torch.float16
        elif dtype == "fp32":
            weight_dtype = torch.float32
        else: 
            weight_dtype = torch.bfloat16
       
        svd_repo = os.path.join(current_node_path, "svd_repo")

        # check model is exits or not,if not auto downlaod
        if SONIC_weigths_path:
            flownet_ckpt = os.path.join(SONIC_weigths_path, "RIFE")
        else:
            flownet_ckpt = os.path.join(Dice_Talk_weigths_path, "RIFE")


        if dice_talk_unet != "none":
            dice_talk_unet = folder_paths.get_full_path("dice_talk", dice_talk_unet)
        else:
            raise  Exception("Please download the model first")
        # load model
        print("***********Load model ***********")
        
        val_noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            svd_repo,
            subfolder="scheduler")

        unet_config_file=os.path.join(svd_repo, "unet")
        unet=convert_cf2diffuser(model.model,unet_config_file,weight_dtype)
        add_ip_adapters(unet, [32, 32], [ip_audio_scale, ip_emo_scale])
        dice_talk_dict=torch.load(dice_talk_unet,weights_only=False, map_location="cpu")
        unet.load_state_dict(dice_talk_dict,strict=True,)
        del dice_talk_dict
        gc_clear()

        pose_guider = PoseGuider(
            conditioning_embedding_channels=320, 
            block_out_channels=(16, 32, 96, 256)
        ).to(device)
        pose_guider_checkpoint_path=os.path.join(Dice_Talk_weigths_path, "pose_guider.pth")
        pose_guider.load_state_dict(
            torch.load(pose_guider_checkpoint_path,weights_only=False, map_location="cpu"),
            strict=True,
        )

        vae_config=os.path.join(svd_repo, "vae/config.json")
        vae_config=OmegaConf.load(vae_config)
      
        pipe = DICE_Talk(device, weight_dtype, vae_config, val_noise_scheduler, unet, flownet_ckpt,pose_guider,use_interframe)

        print("***********Load model done ***********")
        gc_clear()
        return (pipe,weight_dtype)


class Dice_Talk_PreData:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_vision": ("CLIP_VISION",),
                "vae": ("VAE",),
                "audio": ("AUDIO",),
                "image": ("IMAGE",),
                "weight_dtype": ("DTYPE",),
                "emo_files": (os.listdir(os.path.join(current_node_path, "examples/emo")),),
                "min_resolution": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64, "display": "number"}),
                "duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 100000000000.0, "step": 0.1}),
                "expand_ratio": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "retrieval": ("BOOLEAN", {"default": False},),
            }}

    RETURN_TYPES = ("DICETALK_PREDATA",)
    RETURN_NAMES = ("data_dict", )
    FUNCTION = "sampler_main"
    CATEGORY = "Dice_Talk"

    def sampler_main(self, clip_vision,vae, audio, image,weight_dtype,emo_files, min_resolution,duration, expand_ratio,retrieval):
        
        config_file = os.path.join(current_node_path, 'config/inference/dice_talk.yaml')
        config = OmegaConf.load(config_file)

        audio_linear_ckpt = os.path.join(Dice_Talk_weigths_path, "audio_linear.pth")
        emo_model_ckpt = os.path.join(Dice_Talk_weigths_path, "emo_model.pth")
        
        if SONIC_weigths_path:
            yolo_ckpt = os.path.join(SONIC_weigths_path, "yoloface_v5m.pt")
        else:
            yolo_ckpt = os.path.join(Dice_Talk_weigths_path, "yoloface_v5m.pt")

        if not os.path.exists(audio_linear_ckpt) or not os.path.exists(emo_model_ckpt) or not os.path.exists(
                yolo_ckpt):
            raise Exception("Please download the model first")
        
        # init model
        if SONIC_weigths_path:
            whisper_repo = os.path.join(SONIC_weigths_path, "whisper-tiny")
        else:
            whisper_repo = os.path.join(Dice_Talk_weigths_path, "whisper-tiny")

        whisper = WhisperModel.from_pretrained(whisper_repo).to(device).eval()
        whisper.requires_grad_(False)

        feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_repo)

        emo_pe=EmotionModel().to(device)
        audio_linear = AudioProjModel(seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=1024, context_tokens=32).to(device)


        emo_dict = torch.load(emo_model_ckpt,weights_only=False, map_location="cpu")
        audio_linear_dict = torch.load(audio_linear_ckpt,weights_only=False, map_location="cpu")
        emo_pe.load_state_dict(
            emo_dict,
            strict=True,
        )

        audio_linear.load_state_dict(
            audio_linear_dict,
            strict=True,
        )
        del emo_dict, audio_linear_dict
        gc_clear()

        #  处理音频数据
        audio_file_prefix = ''.join(random.choice("0123456789") for _ in range(6))
        audio_path = os.path.join(folder_paths.get_input_directory(), f"audio_{audio_file_prefix}_temp.wav")
        num_frames = audio["waveform"].squeeze(0).shape[1]
        duration_input = num_frames / audio["sample_rate"]

        infer_duration = min(duration,duration_input)
        print(f"Input audio duration is {duration_input} seconds, infer audio duration is: {duration} seconds.")
        # 减少音频数据传递导致的不必要文件存储
        buff = io.BytesIO()
        torchaudio.save(buff, audio["waveform"].squeeze(0), audio["sample_rate"], format="FLAC")

        with open(audio_path, 'wb') as f:
            f.write(buff.getbuffer())
        gc_clear()

        # 处理面部数据
        face_det = AlignImage(device, det_path=yolo_ckpt)
        # 先面部裁切处理
        cv_image = tensor2cv(image)
        face_info = preprocess_face(cv_image, face_det, expand_ratio=expand_ratio)
        if face_info['face_num'] > 0:
            crop_image_pil = cv2pil(crop_face_image(cv_image, face_info['crop_bbox']))

        origin_pil=tensor2pil(image)

        
        emotion_path=os.path.join(current_node_path, "examples/emo",emo_files)
        
        test_data = image_audio_emo_to_tensor(face_det, feature_extractor, origin_pil, audio_path,emotion_path,infer_duration,
                                          limit=MAX_SEED, image_size=min_resolution, area=config.area)

        step = 2
        for k, v in test_data.items():
            if isinstance(v, torch.Tensor):
                test_data[k] = v.unsqueeze(0).to(device).float()
        ref_img = test_data['ref_img']


        
        
        pose_tensor_list,ref_tensor_list,audio_tensor_list,uncond_audio_tensor_list,emotion_tensor_list,uncond_emotion_tensor_list,image_embeds = dice_talk_predata(
            whisper,step,emo_pe,retrieval,test_data,clip_vision,audio_linear,image,device,weight_dtype)
        
        del clip_vision, face_det, whisper
        emo_pe.to("cpu")
        audio_linear.to("cpu")
        gc_clear()

        height, width = ref_img.shape[-2:]

        #print(vae.device,device)
        if vae.device!=device:
            vae.device=device
        img_latent=vae.encode(tensor_upscale(image,width,height)).to(device, dtype=weight_dtype) 
        vae.device=torch.device("cpu")
       
        from comfy.model_management import unload_all_models
        print(unload_all_models())
    
        # bbox_c = face_info['crop_bbox']
        # bbox = [bbox_c[0], bbox_c[1], bbox_c[2] - bbox_c[0], bbox_c[3] - bbox_c[1]]
        return ({"test_data": test_data, "ref_tensor_list": ref_tensor_list, "config": config,"ref_img":ref_img,
                 "image_embeds": image_embeds,"img_latent":img_latent,"vae": vae,"pose_tensor_list": pose_tensor_list,
                 "audio_tensor_list": audio_tensor_list, "uncond_audio_tensor_list": uncond_audio_tensor_list,"uncond_emotion_tensor_list":uncond_emotion_tensor_list,
                 "emotion_tensor_list": emotion_tensor_list},)


class Dice_Talk_Sampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_DICETALK",),
                "data_dict": ("DICETALK_PREDATA",),  # {}
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "inference_steps": ("INT", {"default": 25, "min": 1, "max": 1024, "step": 1, "display": "number"}),
                "ref_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "emo_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 25.0, "min": 5.0, "max": 120.0, "step": 0.5}),
            }}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("image", "fps")
    FUNCTION = "sampler_main"
    CATEGORY = "Dice_Talk"

    def sampler_main(self, model, data_dict, seed, inference_steps, ref_scale,emo_scale, fps):

        print("***********Start infer  ***********")
        # 当前分配的 CUDA 内存
        current_memory = torch.cuda.memory_allocated()
        print(f"Current CUDA memory allocated: {current_memory / 1024**2} MB")

        # 历史最大分配的 CUDA 内存
        max_memory = torch.cuda.max_memory_allocated()
        print(f"Max CUDA memory allocated: {max_memory / 1024**2} MB")

        iamge = model.process(data_dict,
                              fps=fps,
                              inference_steps=inference_steps,
                              ref_scale=ref_scale,
                              emo_scale=emo_scale,
                              seed=seed
                              )
        gc_clear()
        return (iamge.permute(0, 2, 3, 4, 1).squeeze(0), fps)


NODE_CLASS_MAPPINGS = {
    "Dice_Talk_Loader": Dice_Talk_Loader,
    "Dice_Talk_PreData": Dice_Talk_PreData,
    "Dice_Talk_Sampler": Dice_Talk_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Dice_Talk_Loader": "Dice_Talk_Loader",
    "Dice_Talk_PreData": "Dice_Talk_PreData",
    "Dice_Talk_Sampler": "Dice_Talk_Sampler",
}
