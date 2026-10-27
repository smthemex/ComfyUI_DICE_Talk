"""DICE-Talk sliding-window sampler, driven by comfy native apply_model.

Faithful port of DicePipeline.__call__ (any-frame long video):
- karras sigma table identical to diffusers EulerDiscreteScheduler
  (sigma_min 0.002 / sigma_max 700 / rho 7 / final sigma 0, steps_offset=1->karras
  ignores it), timestep = 0.25*ln(sigma), which is exactly comfy
  ModelSamplingContinuousEDM -> native apply_model works as-is.
- The latent buffer is `num_frames + frames_per_batch` long: the trailing window
  worth of frames is a scratch tail that absorbs the wrap-around so the real
  frames never get spliced with frame 0.
- 3-branch CFG: uncond / drop_audio+drop_emo / cond, with
  pred = u + s1*(da-u) + s2*(c-da).
- Per-timestep window sweep with modulo wrap and accumulating shift_offset.
"""
import numpy as np
import torch


def karras_sigmas(num_inference_steps, sigma_min=0.002, sigma_max=700.0, rho=7.0, final_sigma=0.0):
    """diffusers EulerDiscreteScheduler._convert_to_karras + final append."""
    ramp = np.linspace(0, 1, num_inference_steps, dtype=np.float64)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas = np.concatenate([sigmas, [final_sigma]]).astype(np.float32)
    return torch.from_numpy(sigmas)


def get_timesteps_for_strength(num_inference_steps, strength):
    """diffusers get_timesteps: cut the schedule tail for img2img strength."""
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    return t_start  # index into the base sigma table


@torch.no_grad()
def dice_sample(
    model_patcher,
    ref_latent,          # [1, 4, h, w] RAW comfy VAE latent of the reference image
    image_embeds,        # [1, 1024] clip vision image embeds
    audio_list,          # list of [32, 1024], len == num_frames
    uncond_audio_list,
    emo_list,            # list of [32, 1024]
    uncond_emo_list,
    pose_feat,           # [1, 320, h, w] PoseGuider output (frame-independent)
    face_mask,           # [1, 1, H, W] in 0..1
    noise,               # [1, F + frames_per_batch, 4, h, w]
    num_frames,          # real frame count
    fps=12.5,
    motion_bucket_id=8,
    motion_bucket_id_exp=16,
    num_inference_steps=25,
    min_guidance_scale1=3.0,
    max_guidance_scale1=3.0,
    min_guidance_scale2=6.0,
    max_guidance_scale2=6.0,
    overlap=0,
    shift_offset=7,
    frames_per_batch=None,
    i2i_noise_strength=1.0,
    scaling_factor=0.18215,
):
    import comfy.model_management
    import comfy.utils

    base = model_patcher.model
    device = model_patcher.load_device

    # force_full_load skips the lowvram staged-load branch so the whole unet stays
    # resident; load_models_gpu() frees every other model (VAE/CLIP) not kept.
    comfy.model_management.load_models_gpu([model_patcher], force_full_load=True)

    if frames_per_batch is None:
        # official passes frames_per_batch = config.n_sample_frames (25), NOT int(fps).
        frames_per_batch = 25
    dtype = base.get_dtype_inference()

    total = num_frames + frames_per_batch  # latent buffer length (official)

    ref_raw = ref_latent.to(device, torch.float32)            # [1,4,h,w]
    ref_scaled = (ref_raw * scaling_factor).unsqueeze(1)      # init only

    sigmas_all = karras_sigmas(num_inference_steps)
    t_start = get_timesteps_for_strength(num_inference_steps, i2i_noise_strength)
    sigmas = sigmas_all[t_start:].to(device)
    steps = len(sigmas) - 1

    sigma_start = sigmas[0]
    latents_all = (ref_scaled + sigma_start * noise.to(device, torch.float32)).to(dtype)  # [1,total,4,h,w]

    # ---- static conditioning (batch dim 3 = uncond / drop_audio / cond)
    # official: _clip_encode_image() gets num_frames + frames_per_batch, so the clip
    # embedding covers the whole latent buffer including the scratch tail.
    emb = image_embeds.to(device, dtype).unsqueeze(1).repeat(1, total, 1, 1)              # [1,total,1,1024]
    image_cond = torch.cat([torch.zeros_like(emb), emb, emb], dim=0)

    audio_f = torch.stack(audio_list, dim=0).to(device, dtype).unsqueeze(0)                # [1,F,32,1024]
    uaudio_f = torch.stack(uncond_audio_list, dim=0).to(device, dtype).unsqueeze(0)
    emo_f = torch.stack(emo_list, dim=0).to(device, dtype).unsqueeze(0)
    uemo_f = torch.stack(uncond_emo_list, dim=0).to(device, dtype).unsqueeze(0)

    # official appends `frames_per_batch` copies of the FIRST uncond frame to both
    # the conditional and unconditional sequences -> the scratch tail is audio-silent
    pad_a = uaudio_f[:, :1].repeat(1, frames_per_batch, 1, 1)
    pad_e = uemo_f[:, :1].repeat(1, frames_per_batch, 1, 1)
    audio_f = torch.cat([audio_f, pad_a], dim=1)
    uaudio_f = torch.cat([uaudio_f, pad_a], dim=1)
    emo_f = torch.cat([emo_f, pad_e], dim=1)
    uemo_f = torch.cat([uemo_f, pad_e], dim=1)

    audio_cond = torch.cat([uaudio_f, uaudio_f, audio_f], dim=0)                            # [3,total,32,1024]
    emo_cond = torch.cat([uemo_f, uemo_f, emo_f], dim=0)

    # c_concat reference anchor: RAW latent (not scaled), zeros for the uncond branch
    image_latents = ref_raw.clone().unsqueeze(1)
    image_latents = torch.cat([torch.zeros_like(image_latents), image_latents, image_latents], dim=0)
    image_latents = image_latents.repeat(1, total, 1, 1, 1)                                  # [3,total,4,h,w]

    pose = pose_feat.to(device, dtype)                                                       # [1,320,h,w]

    face_mask = face_mask.to(device=device, dtype=dtype)
    if face_mask.ndim == 3:
        face_mask = face_mask.unsqueeze(0)
    face_mask = face_mask[:, :1]

    guidance_scale1 = torch.linspace(min_guidance_scale1, max_guidance_scale1, num_inference_steps).to(device)
    guidance_scale2 = torch.linspace(min_guidance_scale2, max_guidance_scale2, num_inference_steps).to(device)

    added_time_ids = torch.tensor(
        [[float(fps), float(motion_bucket_id), float(motion_bucket_id_exp)]], dtype=torch.float32
    ).to(device).repeat(3, 1)                                                                # [3,3]

    latent_h, latent_w = ref_scaled.shape[-2:]
    shift = 0

    import time
    t_start = time.time()
    n_windows = len(range(0, total, frames_per_batch - overlap))
    print("[DICE] frames=%d buffer=%d windows/step=%d steps=%d dtype=%s"
          % (num_frames, total, n_windows, steps, dtype), flush=True)

    pbar = comfy.utils.ProgressBar(num_inference_steps)
    for i in range(steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        pred_latents = torch.zeros_like(latents_all, dtype=torch.float32)
        counter = torch.zeros((1, total, 1, 1, 1), dtype=torch.float32, device=latents_all.device)

        for index_start in range(0, total, frames_per_batch - overlap):
            index_start = index_start - shift
            idx_list = [(index_start + j) % total for j in range(frames_per_batch)]

            def indice_slice(tensor, idxs):
                return torch.stack([tensor[:, j] for j in idxs], dim=1)

            lat = indice_slice(latents_all, idx_list)                 # [1,fpb,4,h,w]
            c_concat = indice_slice(image_latents, idx_list)          # [3,fpb,4,h,w]
            batch_clip = indice_slice(image_cond, idx_list)           # [3,fpb,1,1024]
            batch_audio = indice_slice(audio_cond, idx_list)          # [3,fpb,32,1024]
            batch_emo = indice_slice(emo_cond, idx_list)              # [3,fpb,32,1024]

            x_in = torch.cat([lat] * 3, dim=0).flatten(0, 1).to(dtype)   # [3*fpb,4,h,w]
            conds = dict(
                c_concat=c_concat.flatten(0, 1),
                added_time_ids=added_time_ids,
                dice_clip=batch_clip,
                dice_audio=batch_audio,
                dice_emo=batch_emo,
                dice_mask=[face_mask, face_mask],
                dice_pose=pose,
            )

            denoised = base.apply_model(x_in, sigma, transformer_options={}, **conds)
            denoised = denoised.view(3, frames_per_batch, 4, latent_h, latent_w).float()

            d_uncond, d_drop, d_cond = denoised[0], denoised[1], denoised[2]
            s1 = guidance_scale1[i]
            s2 = guidance_scale2[i]
            denoised_comb = d_uncond + s1 * (d_drop - d_uncond) + s2 * (d_cond - d_drop)

            lat = lat.float() + (sigma_next - sigma) * (lat.float() - denoised_comb) / sigma
            for j in range(frames_per_batch):
                pred_latents[:, idx_list[j]] += lat[:, j]
                counter[:, idx_list[j]] += 1

        shift = (shift + shift_offset) % frames_per_batch
        latents_all = (pred_latents / counter).to(dtype)
        pbar.update(1)
        if i == 0 or (i + 1) % 5 == 0 or i == steps - 1:
            el = time.time() - t_start
            print("[DICE] step %d/%d  %.1fs elapsed  %.2fs/step"
                  % (i + 1, steps, el, el / (i + 1)), flush=True)

    # keep only the real frames, then scaled -> raw latent space for native VAEDecode
    latents_raw = latents_all[:, :num_frames].float() / scaling_factor
    return latents_raw, sigmas
