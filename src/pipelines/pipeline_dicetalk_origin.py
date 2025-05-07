import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from transformers import CLIPVisionModelWithProjection

from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor, is_compiled_module
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers import (
    AutoencoderKLTemporalDecoder,
    EulerDiscreteScheduler,
)

from ..models.audio_adapter.pose_guider import PoseGuider
from ..models.base.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel

logger = logging.get_logger(__name__)


@dataclass
class Pose2VideoSVDPipelineOutput(BaseOutput):
    r"""
    Output class for zero-shot text-to-video pipeline.

    Args:
        frames (`[List[PIL.Image.Image]`, `np.ndarray`]):
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
    """

    frames: Union[List[PIL.Image.Image], np.ndarray]


class DicePipeline(DiffusionPipeline):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        pose_guider: PoseGuider,
        scheduler: EulerDiscreteScheduler,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            pose_guider=pose_guider,
            scheduler=scheduler,
        )
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True)
        
        self.pose_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )


    def _clip_encode_image(self, image, audio_prompts, uncond_audio_prompts, emo_prompts, uncond_emo_prompts,
                           num_frames, device, num_videos_per_prompt, do_classifier_free_guidance, frames_per_batch):
        dtype = next(self.image_encoder.parameters()).dtype

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)
        
        image_embeddings = image_embeddings.unsqueeze(1).repeat((1, num_frames, 1, 1))
        
        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            
            audio_prompts = torch.stack(audio_prompts, dim=0).to(device=device, dtype=dtype)
            audio_prompts = audio_prompts.unsqueeze(0)
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings, image_embeddings])


            uncond_audio_prompts = torch.stack(uncond_audio_prompts, dim=0).to(device=device, dtype=dtype)
            uncond_audio_prompts = uncond_audio_prompts.unsqueeze(0)

            pad_uncond_audio_prompt = uncond_audio_prompts[:,:1].repeat(1, frames_per_batch, 1, 1)
            audio_prompts = torch.cat([audio_prompts, pad_uncond_audio_prompt], dim=1)
            uncond_audio_prompts = torch.cat([uncond_audio_prompts, pad_uncond_audio_prompt], dim=1)

            emo_prompts = torch.stack(emo_prompts, dim=0).to(device=device, dtype=dtype)
            emo_prompts = emo_prompts.unsqueeze(0)


            uncond_emo_prompts = torch.stack(uncond_emo_prompts, dim=0).to(device=device, dtype=dtype)
            uncond_emo_prompts = uncond_emo_prompts.unsqueeze(0)

            pad_uncond_emo_prompt = uncond_emo_prompts[:,:1].repeat(1, frames_per_batch, 1, 1)
            emo_prompts = torch.cat([emo_prompts, pad_uncond_emo_prompt], dim=1)
            uncond_emo_prompts = torch.cat([uncond_emo_prompts, pad_uncond_emo_prompt], dim=1)


            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            audio_prompts = torch.cat([uncond_audio_prompts, uncond_audio_prompts, audio_prompts])
            emo_prompts = torch.cat([uncond_emo_prompts, uncond_emo_prompts, emo_prompts])
            # import pdb;pdb.set_trace()

        return image_embeddings, audio_prompts, emo_prompts

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.mode()
        
        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents, image_latents])

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        return image_latents
    
    def _get_add_time_ids(
        self,
        fps,
        motion_bucket_id,
        noise_aug_strength,
        dtype,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        add_time_ids = [fps, motion_bucket_id, noise_aug_strength]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids, add_time_ids])

        return add_time_ids
    
    def decode_latents(self, latents, num_frames, decode_chunk_size=14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def check_inputs(self, image, height, width):
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                f" {type(image)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    def prepare_latents(
        self,
        batch_size,
        num_frames,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        ref_image_latents=None,
        timestep=None
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            noise = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        if timestep is not None:
            init_latents = ref_image_latents.unsqueeze(1)
            latents = self.scheduler.add_noise(init_latents, noise, timestep)
        else:
            latents = noise * self.scheduler.init_noise_sigma
        return latents
    
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    @property
    def guidance_scale1(self):
        return self._guidance_scale1
    
    @property
    def guidance_scale2(self):
        return self._guidance_scale2

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return True

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    def __call__(
        self,
        ref_image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        clip_image: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        pose_images: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        audio_prompts: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        uncond_audio_prompts: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        emo_prompts: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        uncond_emo_prompts: Union[PIL.Image.Image, List[PIL.Image.Image], torch.FloatTensor],
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale1=1.0, # 1.0,
        max_guidance_scale1=3.0,
        min_guidance_scale2=1.0, # 1.0,
        max_guidance_scale2=3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        motion_bucket_id_exp: int = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        overlap=7,
        shift_offset=3,
        frames_per_batch=14,
        i2i_noise_strength=1.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`int`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor


        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(ref_image, height, width)

        # 2. Define call parameters
        if isinstance(ref_image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(ref_image, list):
            batch_size = len(ref_image)
        else:
            batch_size = ref_image.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = True

        # 3. Prepare clip image embeds
        image_embeddings, audio_prompts, emo_prompts = self._clip_encode_image(
            clip_image, 
            audio_prompts,
            uncond_audio_prompts,
            emo_prompts,
            uncond_emo_prompts,
            num_frames + frames_per_batch,
            device, 
            num_videos_per_prompt, 
            do_classifier_free_guidance,
            frames_per_batch)        

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        # fps = fps - 1

        # 4. Encode input image using VAE
        needs_upcasting = (self.vae.dtype == torch.float16 or self.vae.dtype == torch.bfloat16) and self.vae.config.force_upcast
        vae_dtype = self.vae.dtype
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
        
        # Prepare ref image latents
        ref_image_tensor = ref_image.to(
            dtype=self.vae.dtype, device=self.vae.device
        )
 
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)

        noise = randn_tensor(
            ref_image_tensor.shape, 
            generator=generator, 
            device=self.vae.device, 
            dtype=self.vae.dtype)
        
        ref_image_tensor = ref_image_tensor + noise_aug_strength * noise

        image_latents = self._encode_vae_image(
            ref_image_tensor,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)
        ref_image_latents = ref_image_latents.to(image_embeddings.dtype)
        
        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=vae_dtype)
        
        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames + frames_per_batch, 1, 1, 1)        

        
         # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            motion_bucket_id_exp,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device, dtype=self.unet.dtype)
        # import pdb;pdb.set_trace()

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, i2i_noise_strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_videos_per_prompt)


        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames + frames_per_batch,
            num_channels_latents,
            height,
            width,
            image_embeddings.dtype,
            device,
            generator,
            latents,
            ref_image_latents,
            timestep=latent_timestep
        )

        # Prepare a list of pose condition images
        pose_cond_tensor_list = []
        for pose_image in pose_images:
            pose_cond_tensor = pose_image.unsqueeze(0)
            pose_cond_tensor = pose_cond_tensor.unsqueeze(2)  # (bs, c, 1, h, w)
            pose_cond_tensor_list.append(pose_cond_tensor)
        
        for _ in range(frames_per_batch):
            pose_cond_tensor_list.append(pose_cond_tensor)

        pose_cond_tensor = torch.cat(pose_cond_tensor_list, dim=2)  # (bs, c, t, h, w)
        pose_cond_tensor = pose_cond_tensor.to(
            device=device, dtype=self.pose_guider.dtype
        )
        face_mask = pose_cond_tensor[0, :1, :1]
        # print(pose_cond_tensor.shape)

        pose_fea = self.pose_guider(pose_cond_tensor).transpose(
            1, 2
        ) # (bs, f, c, H, W)
       
        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(
            min_guidance_scale1, 
            max_guidance_scale1, 
            num_inference_steps)
        guidance_scale1 = guidance_scale.to(device, latents.dtype)

        guidance_scale = torch.linspace(
            min_guidance_scale2, 
            max_guidance_scale2, 
            num_inference_steps)
        guidance_scale2 = guidance_scale.to(device, latents.dtype)

        # print(guidance_scale)
        
        self._guidance_scale1 = guidance_scale1
        self._guidance_scale2 = guidance_scale2

        # 8. Denoising loop
        latents_all = latents # for any-frame generation

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        shift = 0
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # init 
                pred_latents = torch.zeros_like(
                    latents_all,
                    dtype=self.unet.dtype,
                )
                counter = torch.zeros(
                    (latents_all.shape[0], num_frames+frames_per_batch, 1, 1, 1),
                    dtype=self.unet.dtype,
                ).to(device=latents_all.device)
                # print(t)

                for batch, index_start in enumerate(range(0, num_frames+frames_per_batch, frames_per_batch - overlap)):
                    self.scheduler._step_index = None
                    index_start -= shift
                    # print(index_start)
                    def indice_slice(tensor, idx_list):
                        tensor_list = []
                        for idx in idx_list:
                            idx = idx % tensor.shape[1]
                            tensor_list.append(tensor[:,idx])
                        return torch.stack(tensor_list, 1)
                    idx_list = list(range(index_start, index_start+frames_per_batch))
                    latents = indice_slice(latents_all, idx_list)
                    pose_cond_fea = indice_slice(pose_fea, idx_list)
                    image_latents_input = indice_slice(image_latents, idx_list)
                    batch_image_embeddings = indice_slice(image_embeddings, idx_list)
                    batch_audio_prompts = indice_slice(audio_prompts, idx_list)
                    # import pdb;pdb.set_trace()
                    
                    batch_emo_prompts = indice_slice(emo_prompts, idx_list)
                    # import pdb;pdb.set_trace()

                    cross_attention_kwargs = {'ip_adapter_masks': [face_mask]*2}
                    latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    pose_cond_fea = pose_cond_fea.repeat(3 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    # latent_model_input = torch.cat([latents] * 4) if do_classifier_free_guidance else latents
                    # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    # pose_cond_fea = pose_cond_fea.repeat(4 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
                    # import ipdb;ipdb.set_trace()
                    # Concatenate image_latents over channels dimention
                    latent_model_input = torch.cat([
                        latent_model_input, 
                        image_latents_input], dim=2)
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=(batch_image_embeddings.flatten(0,1), [batch_audio_prompts.flatten(0,1)] + [batch_emo_prompts.flatten(0,1)]),
                        spatial_condition=pose_cond_fea,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]        
                    # perform guidance
                    # import ipdb;ipdb.set_trace()
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_drop_audio, noise_pred_cond = noise_pred.chunk(3)
                        noise_pred = noise_pred_uncond + self.guidance_scale1[i] * (noise_pred_drop_audio - noise_pred_uncond) \
                            + self.guidance_scale2[i] * (noise_pred_cond - noise_pred_drop_audio)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t.to(self.unet.dtype), latents).prev_sample

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)

                    # if batch == 0:
                    for iii in range(frames_per_batch):
                        p = (index_start + iii) % pred_latents.shape[1]
                        pred_latents[:, p] += latents[:, iii]
                        counter[:, p] += 1
                shift += shift_offset
                shift = shift % frames_per_batch

                pred_latents  = pred_latents / counter
                latents_all = pred_latents

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        latents = latents_all
        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=vae_dtype)
            frames = self.decode_latents(latents, num_frames + frames_per_batch, decode_chunk_size)[:,:,:num_frames]
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames
        return Pose2VideoSVDPipelineOutput(frames=frames)


