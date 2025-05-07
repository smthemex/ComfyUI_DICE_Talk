import os
import torch
from tqdm import tqdm

from .src.utils.util import  seed_everything
from .src.dataset.test_preprocess import process_bbox
from .src.pipelines.pipeline_dicetalk import DicePipeline

from .src.utils.RIFE.RIFE_HDv3 import RIFEModel



BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def dice_talk_predata(wav_enc,step,emo_pe,retrieval,batch,image_encoder,audio_pe,clip_img,device,weight_dtype):
 
    image_embeds=image_encoder.encode_image(clip_img)["image_embeds"] #torch.Size([1, 1024])

    if device!=torch.device("cpu"):
        image_embeds=image_embeds.clone().detach().to(device, dtype=weight_dtype) # mps or cuda
    else:
        image_embeds=image_embeds.to(device, dtype=weight_dtype) 

    ref_img = batch['ref_img']
    audio_feature = batch['audio_feature']
    audio_len = batch['audio_len']
    emo_prior = batch['emo_feature']


    window = 3000
    audio_prompts = []
    for i in range(0, audio_feature.shape[-1], window):
        audio_prompt = wav_enc.encoder(audio_feature[:,:,i:i+window], output_hidden_states=True).hidden_states
        audio_prompt = torch.stack(audio_prompt, dim=2)
        audio_prompts.append(audio_prompt)
    audio_prompts = torch.cat(audio_prompts, dim=1)
    audio_prompts = audio_prompts[:,:audio_len*2]

    audio_prompts = torch.cat([torch.zeros_like(audio_prompts[:,:4]), audio_prompts, torch.zeros_like(audio_prompts[:,:6])], 1)

    pose_tensor_list = []
    ref_tensor_list = []
    audio_tensor_list = []
    uncond_audio_tensor_list = []
    emotion_tensor_list = []
    uncond_emotion_tensor_list = []

    
    for i in tqdm(range(audio_len//step)):

        pixel_values_pose = batch["face_mask"]

        audio_clip = audio_prompts[:,i*2*step:i*2*step+10].unsqueeze(0)
        cond_audio_clip = audio_pe(audio_clip).squeeze(0)
        uncond_audio_clip = audio_pe(torch.zeros_like(audio_clip)).squeeze(0)


        new_emo_hidden_states = emo_pe(emo_prior, retrieval=retrieval)[0].squeeze(0)
        new_uncond_emo_hidden_states = emo_pe(torch.zeros_like(emo_prior), retrieval=retrieval)[0].squeeze(0)


        pose_tensor_list.append(pixel_values_pose[0])
        ref_tensor_list.append(ref_img[0])
        audio_tensor_list.append(cond_audio_clip[0])
        uncond_audio_tensor_list.append(uncond_audio_clip[0])

        emotion_tensor_list.append(new_emo_hidden_states[0])
        uncond_emotion_tensor_list.append(new_uncond_emo_hidden_states[0])


    return pose_tensor_list,ref_tensor_list,audio_tensor_list,uncond_audio_tensor_list,emotion_tensor_list,uncond_emotion_tensor_list,image_embeds


def preprocess_face(face_image,face_det, expand_ratio=1.0):
    #face_image = cv2.imread(image_path)
    h, w = face_image.shape[:2]
    _, _, bboxes = face_det(face_image, maxface=True)
    face_num = len(bboxes)
    bbox = []
    if face_num > 0:
        x1, y1, ww, hh = bboxes[0]
        x2, y2 = x1 + ww, y1 + hh
        bbox = x1, y1, x2, y2
        bbox_s = process_bbox(bbox, expand_radio=expand_ratio, height=h, width=w)
    else:
        raise ValueError('No face detected')
    return {
        'face_num': face_num,
        'crop_bbox': bbox_s,
    }
    
def crop_face_image(face_image,crop_bbox): #NEED CHECK
    crop_image = face_image[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
    return crop_image

def decode_latents_(latents,vae,device, decode_chunk_size=14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / 0.18215 * latents
        vae.device = device
        
        # forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        # accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            #num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            #decode_kwargs = {}
            # if accepts_num_frames:
            #     # we only pass num_frames_in if it's expected
            #     decode_kwargs["num_frames"] = num_frames_in

            frame = vae.decode(latents[i : i + decode_chunk_size])
            frames.append(frame.cpu())
        frames = torch.cat(frames, dim=0) # [50, 512, 512, 3]

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        #frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        frames=frames.unsqueeze(0).permute(0, 4, 1, 2, 3)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames


def test(
    pipe,
    config,
    fps,
    width,
    height,
    data_dict,
    device,
    weight_dtype,
):  
   

    pipe.to(device=device, dtype=weight_dtype)

    video = pipe(
        data_dict["ref_img"],
        data_dict["image_embeds"],
        data_dict["pose_tensor_list"],
        data_dict["audio_tensor_list"],
        data_dict["uncond_audio_tensor_list"],
        data_dict["emotion_tensor_list"],
        data_dict["uncond_emotion_tensor_list"],
        height=height,
        width=width,
        num_frames=len(data_dict["pose_tensor_list"]),
        decode_chunk_size=config.decode_chunk_size,
        motion_bucket_id=config.motion_bucket_id,
        motion_bucket_id_exp=config.motion_bucket_id_exp,
        fps=fps,
        noise_aug_strength=config.noise_aug_strength,
        min_guidance_scale1=config.min_appearance_guidance_scale, # 1.0,
        max_guidance_scale1=config.max_appearance_guidance_scale,
        min_guidance_scale2=config.audio_guidance_scale, # 1.0,
        max_guidance_scale2=config.audio_guidance_scale,
        overlap=config.overlap,
        shift_offset=config.shift_offset,
        frames_per_batch=config.n_sample_frames,
        num_inference_steps=config.num_inference_steps,
        i2i_noise_strength=config.i2i_noise_strength,
        img_latent=data_dict["img_latent"],
    ).frames

    # Concat it with pose tensor
    # pose_tensor = torch.stack(pose_tensor_list,1).unsqueeze(0)
    pipe.to(device=torch.device("cpu"))

    video=decode_latents_(video, data_dict["vae"],device, decode_chunk_size=14) # torch.Size([1, 3, 250, 512, 512])
    # video = (video*0.5 + 0.5).clamp(0, 1)
    # video = torch.cat([video.to(device="cuda")], dim=0).cpu()

    return video


class DICE_Talk():
    #config_file = os.path.join(BASE_DIR, 'config/inference/dice_talk.yaml')
    #config = OmegaConf.load(config_file)

    def __init__(self, 
                 device,
                 weight_dtype,
                 vae_config, val_noise_scheduler, unet, flownet_ckpt,pose_guider,
                 use_interframe=True,
                 ):
        
        #config = self.config
        #config.use_interframe = enable_interpolate_frame
        self.use_interframe = use_interframe
        #device = 'cuda:{}'.format(device_id) if device_id > -1 else 'cpu'
        self.weight_dtype = weight_dtype
        #config.pretrained_model_name_or_path = os.path.join(BASE_DIR, config.pretrained_model_name_or_path)

        if self.use_interframe:
            rife = RIFEModel(device=device)
            rife.load_model(flownet_ckpt)
            self.rife = rife

        # image_encoder.to(weight_dtype)
        # vae.to(weight_dtype)
        unet.to(weight_dtype)

        self.pipe = DicePipeline(
            unet=unet,
            vae_config=vae_config,
            pose_guider=pose_guider,
            scheduler=val_noise_scheduler,
        )

        self.device = device

        print('init done')


    

    @torch.no_grad()
    def process(self,data_dict,fps,
                inference_steps=25,
                ref_scale=None,
                emo_scale=None,
                seed=None):
        
        config = data_dict["config"]
        test_data = data_dict["test_data"]

        # specific parameters
        if seed:
            config.seed = seed

        config.num_inference_steps = inference_steps

        if ref_scale is not None:
            config.min_appearance_guidance_scale = ref_scale
            config.max_appearance_guidance_scale = ref_scale
        if emo_scale is not None:
            config.audio_guidance_scale = emo_scale


        seed_everything(seed)

        height, width = test_data['ref_img'].shape[-2:]

 

        video = test(
            self.pipe,
            config,
            fps,
            width=width,
            height=height,
            data_dict=data_dict,
            device=self.device,
            weight_dtype=self.weight_dtype,
            )

        if self.use_interframe:
            rife = self.rife
            out = video.to(self.device)
            results = []
            video_len = out.shape[2]
            for idx in tqdm(range(video_len-1), ncols=0):
                I1 = out[:, :, idx]
                I2 = out[:, :, idx+1]
                middle = rife.inference(I1, I2).clamp(0, 1).detach()
                results.append(out[:, :, idx])
                results.append(middle)
            results.append(out[:, :, video_len-1])
            video = torch.stack(results, 2).cpu()
        
        # save_videos_grid(video, video_path, n_rows=video.shape[0], fps=config.fps * 2 if config.use_interframe else config.fps)
        # ffmpeg_command = f'ffmpeg -i "{video_path}" -i "{audio_path}" -s {resolution} -vcodec libx264 -acodec aac -crf 18 -shortest -y "{audio_video_path}"'
        # os.system(ffmpeg_command)
        # os.remove(video_path)  # Use os.remove instead of rm for Windows compatibility
        
        return video
        
