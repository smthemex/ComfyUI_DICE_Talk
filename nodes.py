"""ComfyUI nodes for DICE-Talk (native refactor).

Chain: DICE_Native_Loader -> DICE_Native_PreData -> DICE_Native_Sampler -> native VAEDecode
       -> (optional) DICE_Native_Rife
No diffusers / transformers imports anywhere: only torch / torchaudio / numpy / PIL.
"""
import os

import numpy as np
import torch

import folder_paths

from .dice_core.audio_front import WhisperEncoder, log_mel_spectrogram
from .dice_core.dice_model import load_dice_talk_model
from .dice_core.emo import AudioProjModel, EmotionModel
from .dice_core.face_align.align import AlignImage
from .dice_core.pose_guider import PoseGuider
from .dice_core.rife.RIFE_HDv3 import RIFEModel
from .dice_core.sampler import dice_sample

MAX_SEED = np.iinfo(np.int32).max

_models_dir = os.path.join(folder_paths.models_dir, "sonic")
if not os.path.exists(_models_dir):
    os.makedirs(_models_dir)
folder_paths.add_model_folder_path("sonic", _models_dir)

_cache = {}


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pick_dice_unet():
    """dice_unet.pth lives in models/sonic alongside sonic's unet.pth."""
    cands = [f for f in folder_paths.get_filename_list("sonic") if "dice" in f.lower()]
    if cands:
        return cands[0]
    return None


def _get_face_detector():
    if "face_det" in _cache:
        return _cache["face_det"]
    det = AlignImage(_device(), det_path=os.path.join(_models_dir, "yoloface_v5m.pt"))
    _cache["face_det"] = det
    return det


def _get_rife():
    if "rife" in _cache:
        return _cache["rife"]
    rife = RIFEModel(device=_device())
    # RIFEModel.load_model() appends "flownet.pkl" internally -> pass the DIRECTORY.
    rife.load_model(os.path.join(_models_dir, "RIFE"))
    _cache["rife"] = rife
    return _cache["rife"]


def _get_audio_front():
    """whisper encoder + audio projection + emotion model (DICE weights).

    Kept in fp32 like the official code: whisper / audio_linear / emo_model are NOT
    part of the pipe there, so `pipe.to(dtype)` never touches them. Their outputs are
    cast to the model dtype at the sampler boundary.
    """
    key = "audio_front"
    if key in _cache:
        return _cache[key]
    dev = _device()

    whisper = WhisperEncoder().to(dev).eval()
    whisper.load_hf_safetensors(os.path.join(_models_dir, "whisper-tiny", "model.safetensors"))
    whisper.requires_grad_(False)

    audio_linear = AudioProjModel(seq_len=10, blocks=5, channels=384, intermediate_dim=1024,
                                  output_dim=1024, context_tokens=32).to(dev).eval()
    sd = torch.load(os.path.join(_models_dir, "audio_linear.pth"), map_location="cpu", weights_only=True)
    audio_linear.load_state_dict(sd, strict=True)

    emo_model = EmotionModel().to(dev).eval()
    sd = torch.load(os.path.join(_models_dir, "emo_model.pth"), map_location="cpu", weights_only=True)
    emo_model.load_state_dict(sd, strict=True)

    _cache[key] = (whisper, audio_linear, emo_model)
    return _cache[key]


def _get_pose_guider(dtype):
    key = "pose_guider_" + str(dtype)
    if key in _cache:
        return _cache[key]
    pg = PoseGuider(conditioning_embedding_channels=320, block_out_channels=(16, 32, 96, 256)).to(_device()).eval()
    sd = torch.load(os.path.join(_models_dir, "pose_guider.pth"), map_location="cpu", weights_only=True)
    pg.load_state_dict(sd, strict=True)
    pg.to(dtype)
    _cache[key] = pg
    return _cache[key]


def encode_audio_features(waveform, sample_rate, duration):
    """waveform [B, T, C] -> (mel-chunked whisper hidden states [1,T,5,384], audio_len)."""
    import torchaudio.functional as AF

    wav = waveform
    if wav.ndim == 3:      # [B, T, C]
        wav = wav.mean(dim=0)
    if wav.ndim == 2:      # [T, C] -> mono
        wav = wav.mean(dim=0)
    if sample_rate != 16000:
        wav = AF.resample(wav.float(), sample_rate, 16000)
    # duration <= 0 means "use the whole audio" (no truncation)
    if duration and duration > 0:
        max_samples = int(duration * 16000)
        if wav.shape[0] > max_samples:
            wav = wav[:max_samples]

    device = _device()
    whisper, _, _ = _get_audio_front()

    window = 750 * 640  # 480000 samples = 30 s
    parts = []
    with torch.no_grad():
        for i in range(0, wav.shape[0], window):
            mel = log_mel_spectrogram(wav[i:i + window].to(device))
            hs, _ = whisper(mel)                    # tuple of layers+1, each [1,1500,384]
            parts.append(torch.stack(hs, dim=2))    # [1,1500,5,384]

    audio_prompts = torch.cat(parts, dim=1)
    audio_len = wav.shape[0] // 640
    return audio_prompts, audio_len


def _dice_unet_files():
    """Only .pth files from models/sonic, with dice_unet first (the intended pick)."""
    files = [f for f in folder_paths.get_filename_list("sonic") if f.lower().endswith(".pth")]
    files.sort(key=lambda f: (0 if "dice" in f.lower() else 1, f))
    return files


class DICENativeLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "svd_checkpoint": (folder_paths.get_filename_list("checkpoints"),),
                "dice_unet": (_dice_unet_files(),),
                "ip_audio_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "ip_emo_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "dtype": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("MODEL", "VAE", "CLIP_VISION", "DTYPE")
    RETURN_NAMES = ("model", "vae", "clip_vision", "dtype")
    FUNCTION = "load"
    CATEGORY = "DICE_Talk_Native"

    def load(self, svd_checkpoint, dice_unet, ip_audio_scale, ip_emo_scale, dtype):
        dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        ckpt = folder_paths.get_full_path("checkpoints", svd_checkpoint)
        unet = folder_paths.get_full_path("sonic", dice_unet)
        patcher, vae, clip_vision = load_dice_talk_model(
            ckpt, unet, ip_audio_scale=ip_audio_scale, ip_emo_scale=ip_emo_scale, dtype=dt)
        return (patcher, vae, clip_vision, dt)


class DICENativePreData:
    @classmethod
    def INPUT_TYPES(s):
        emo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotions")
        emos = sorted(f[:-4] for f in os.listdir(emo_dir) if f.endswith(".npy")) if os.path.isdir(emo_dir) else ["neutral"]
        return {
            "required": {
                "clip_vision": ("CLIP_VISION",),
                "vae": ("VAE",),
                "audio": ("AUDIO",),
                "image": ("IMAGE",),
                "dtype": ("DTYPE",),
                "emotion": (emos,),
                "min_resolution": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64}),
                "duration": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100000.0, "step": 0.1}),
                "retrieval": ("BOOLEAN", {"default": False}),
                "area": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.5, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("DICE_NATIVE_DATA",)
    FUNCTION = "predata"
    CATEGORY = "DICE_Talk_Native"

    def predata(self, clip_vision, vae, audio, image, dtype, emotion, min_resolution, duration,
                retrieval, area):
        dev = _device()
        dt = dtype
        image0 = image[:1]  # [1,H,W,C] 0..1

        # ---- face detect. Official uses the box ONLY for the mask + pose guider input
        np_img = (image0[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)  # RGB
        face_det = _get_face_detector()
        _, _, bboxes = face_det(np_img[:, :, ::-1], maxface=True)  # det expects BGR
        if len(bboxes) == 0:
            raise Exception("DICE-Talk: no face detected in the input image")
        x1, y1, ww, hh = [float(v) for v in bboxes[0]]
        x2, y2 = x1 + ww, y1 + hh
        h, w = np_img.shape[:2]

        face_frac = hh / float(h)
        if face_frac < 0.15:
            print("[DICE Native] WARNING: detected face occupies only {:.0%} of the image height; "
                  "DICE-Talk is trained on head-and-shoulders human portraits.".format(face_frac))

        # ---- official sizing: scale = image_size / min(w,h), rounded to 64 multiples
        scale = min_resolution / min(h, w)
        new_w = max(64, round(w * scale / 64) * 64)
        new_h = max(64, round(h * scale / 64) * 64)

        # ---- face mask (area-expanded box on the original image, then LANCZOS)
        mask_img = np.zeros_like(np_img)
        aw, ah = (x2 - x1) * area, (y2 - y1) * area
        cx, cy = int((x2 + x1) // 2), int((y2 + y1) // 2)
        mx1, my1 = max(cx - int(aw // 2), 0), max(cy - int(ah // 2), 0)
        mx2, my2 = min(cx + int(aw // 2), w), min(cy + int(ah // 2), h)
        mask_img[my1:my2, mx1:mx2] = 255

        from PIL import Image as _Image
        mask_pil = _Image.fromarray(mask_img).resize((new_w, new_h), _Image.LANCZOS)
        mask3 = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0).permute(2, 0, 1)  # [3,H,W]
        face_mask = mask3[:1].unsqueeze(0)   # [1,1,H,W] for the IP adapters

        # ---- reference image = the WHOLE image, LANCZOS to 64-multiple
        ref_pil = _Image.fromarray(np_img).resize((new_w, new_h), _Image.LANCZOS)
        ref01 = torch.from_numpy(np.array(ref_pil).astype(np.float32) / 255.0)  # [h,w,3]

        # ---- pose guider feature: input is the 3-channel mask, constant over frames,
        #      and PoseGuider is a per-frame 2D conv stack -> compute once and broadcast.
        pg = _get_pose_guider(dt)
        with torch.no_grad():
            pose_in = mask3.to(dev, dt).unsqueeze(0).unsqueeze(2)  # [1,3,1,H,W]
            pose_feat = pg(pose_in).transpose(1, 2)[:, 0]          # [1,320,h,w]

        # ---- audio
        audio_prompts, audio_len = encode_audio_features(audio["waveform"], audio["sample_rate"], duration)
        step = 2
        P = audio_prompts[:, : audio_len * step]
        P = torch.cat([torch.zeros_like(P[:, :4]), P, torch.zeros_like(P[:, :6])], dim=1)  # [1,T+10,5,384]

        whisper, audio_linear, emo_model = _get_audio_front()
        emo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotions", emotion + ".npy")
        emo_prior = torch.from_numpy(
            np.load(emo_path, allow_pickle=True).item()["mu"]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # the emo model runs in fp32 (official never casts it) -> keep the prior fp32
        emo_prior = emo_prior.to(dev, torch.float32)   # [1,1,1,1,256] (5D as official expects)

        with torch.no_grad():
            image_embeds = clip_vision.encode_image(image0).image_embeds.float()  # [1,1024]

            cond_emo = emo_model(emo_prior, retrieval=retrieval)[0].squeeze(0)[0].float()       # [32,1024]
            ucond_emo = emo_model(torch.zeros_like(emo_prior), retrieval=retrieval)[0].squeeze(0)[0].float()

            audio_tensor_list, uncond_audio_tensor_list = [], []
            for i in range(audio_len // step):
                clip = P[:, i * 2 * step: i * 2 * step + 10].unsqueeze(0)     # [1,1,10,5,384]
                audio_tensor_list.append(audio_linear(clip).squeeze(0)[0].float())
                uncond_audio_tensor_list.append(audio_linear(torch.zeros_like(clip)).squeeze(0)[0].float())

        num_frames = len(audio_tensor_list)
        if num_frames < 1:
            raise Exception("DICE-Talk: audio too short")

        # ---- conditioning latent (RAW). Official encodes the whole resized image.
        with torch.no_grad():
            ref_latent = vae.encode(ref01.unsqueeze(0).to(device=dev, dtype=torch.float32))

        data = {
            "ref_latent": ref_latent.float().cpu(),
            "image_embeds": image_embeds.cpu(),
            "audio_list": [a.cpu() for a in audio_tensor_list],
            "uncond_audio_list": [a.cpu() for a in uncond_audio_tensor_list],
            "emo": cond_emo.cpu(),
            "uncond_emo": ucond_emo.cpu(),
            "pose_feat": pose_feat.float().cpu(),
            "face_mask": face_mask.float().cpu(),
            "num_frames": num_frames,
            "ref_size": (new_h, new_w),
        }
        return (data,)


class DICENativeSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "dice_data": ("DICE_NATIVE_DATA",),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 100}),
                "appearance_guidance": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "emotion_guidance": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 12.5, "min": 5.0, "max": 120.0, "step": 0.5}),
                "motion_bucket_id": ("INT", {"default": 8, "min": 1, "max": 255}),
                "motion_bucket_id_exp": ("INT", {"default": 16, "min": 1, "max": 255}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                # official passes frames_per_batch = n_sample_frames (25)
                "frames_per_batch": ("INT", {"default": 25, "min": 1, "max": 2048}),
                "overlap": ("INT", {"default": 0, "min": 0, "max": 24}),
                "shift_offset": ("INT", {"default": 7, "min": 0, "max": 24}),
            },
        }

    RETURN_TYPES = ("LATENT", "FLOAT", "INT")
    RETURN_NAMES = ("latent", "fps", "num_frames")
    FUNCTION = "sample"
    CATEGORY = "DICE_Talk_Native"

    def sample(self, model, dice_data, seed, steps, appearance_guidance, emotion_guidance,
               fps, motion_bucket_id, motion_bucket_id_exp, strength,
               frames_per_batch=None, overlap=0, shift_offset=7):
        num_frames = dice_data["num_frames"]
        latent_h, latent_w = dice_data["ref_latent"].shape[-2:]  # raw latent spatial size
        dev = _device()
        fpb = frames_per_batch if frames_per_batch else 25

        # Official RNG: global torch.manual_seed + randn on CUDA in fp16. The wrapper path
        # passes img_latent, so the pipeline's noise_aug draw is skipped -> the latent noise
        # is the FIRST draw of the stream.
        torch.manual_seed(seed)
        nptype = torch.float16 if dev.type == "cuda" else torch.float32
        noise = torch.randn((1, num_frames + fpb, 4, latent_h, latent_w), dtype=nptype, device=dev)

        latents_raw, _ = dice_sample(
            model,
            ref_latent=dice_data["ref_latent"],
            image_embeds=dice_data["image_embeds"],
            audio_list=dice_data["audio_list"],
            uncond_audio_list=dice_data["uncond_audio_list"],
            emo_list=[dice_data["emo"]] * num_frames,
            uncond_emo_list=[dice_data["uncond_emo"]] * num_frames,
            pose_feat=dice_data["pose_feat"],
            face_mask=dice_data["face_mask"],
            noise=noise,
            num_frames=num_frames,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            motion_bucket_id_exp=motion_bucket_id_exp,
            num_inference_steps=steps,
            min_guidance_scale1=appearance_guidance,
            max_guidance_scale1=appearance_guidance,
            min_guidance_scale2=emotion_guidance,
            max_guidance_scale2=emotion_guidance,
            overlap=overlap,
            shift_offset=shift_offset,
            frames_per_batch=fpb,
            i2i_noise_strength=strength,
        )
        latent_out = latents_raw.flatten(0, 1)  # [F,4,h,w] raw -> native VAEDecode
        return ({"samples": latent_out.cpu()}, fps, num_frames)


class DICENativeRife:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {"default": 12.5, "min": 1.0, "max": 120.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("image", "fps")
    FUNCTION = "interpolate"
    CATEGORY = "DICE_Talk_Native"

    def interpolate(self, images, fps):
        rife = _get_rife()
        frames = images.to(_device()).movedim(-1, 1)  # [F,C,H,W]
        results = []
        n = frames.shape[0]
        with torch.no_grad():
            for idx in range(n - 1):
                I1 = frames[idx: idx + 1]
                I2 = frames[idx + 1: idx + 2]
                middle = rife.inference(I1, I2).clamp(0, 1).detach()  # [1,C,H,W]
                results.append(frames[idx: idx + 1])
                results.append(middle)
            results.append(frames[n - 1: n])
        out = torch.cat(results, dim=0).movedim(1, -1).cpu()  # [2F-1,H,W,C]
        return (out, fps * 2.0)


NODE_CLASS_MAPPINGS = {
    "DICENativeLoader": DICENativeLoader,
    "DICENativePreData": DICENativePreData,
    "DICENativeSampler": DICENativeSampler,
    "DICENativeRife": DICENativeRife,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DICENativeLoader": "DICE-Talk Native Loader",
    "DICENativePreData": "DICE-Talk Native PreData",
    "DICENativeSampler": "DICE-Talk Native Sampler",
    "DICENativeRife": "DICE-Talk Native RIFE (interpolate x2)",
}
