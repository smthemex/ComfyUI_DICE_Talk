import importlib
import os
import os.path as osp
import shutil
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
# import skvideo
# import skvideo.io
skvideo=None
import cv2


class VideoUtils(object):
    def __init__(self, video_path=None, output_video_path=None, bit_rate='origin', fps=25):
        if video_path is not None:
            meta_data = skvideo.io.ffprobe(video_path)
            # avg_frame_rate = meta_data['video']['@r_frame_rate']
            # a, b = avg_frame_rate.split('/')
            # fps = float(a) / float(b)
            # fps = 25
            codec_name = 'libx264'
            # codec_name = meta_data['video'].get('@codec_name')
            # if codec_name=='hevc':
            #     codec_name='h264'
            # profile = meta_data['video'].get('@profile')
            color_space = meta_data['video'].get('@color_space')
            color_transfer = meta_data['video'].get('@color_transfer')
            color_primaries = meta_data['video'].get('@color_primaries')
            color_range = meta_data['video'].get('@color_range')
            pix_fmt = meta_data['video'].get('@pix_fmt')
            if bit_rate=='origin':
                bit_rate = meta_data['video'].get('@bit_rate')
            else:
                bit_rate=None
            if pix_fmt is None:
                pix_fmt = 'yuv420p'

            reader_output_dict = {'-r': str(fps)}
            writer_input_dict = {'-r': str(fps)}
            writer_output_dict = {'-pix_fmt': pix_fmt, '-r': str(fps), '-vcodec':str(codec_name)}
            # if bit_rate is not None:
            #     writer_output_dict['-b:v'] = bit_rate
            writer_output_dict['-crf'] = '17'

            # if video has alpha channel, convert to bgra, uint16 to process
            if pix_fmt.startswith('yuva'):
                writer_input_dict['-pix_fmt'] = 'bgra64le'
                reader_output_dict['-pix_fmt'] = 'bgra64le'
            elif pix_fmt.endswith('le'):
                writer_input_dict['-pix_fmt'] = 'bgr48le'
                reader_output_dict['-pix_fmt'] = 'bgr48le'
            else:
                writer_input_dict['-pix_fmt'] = 'bgr24'
                reader_output_dict['-pix_fmt'] = 'bgr24'

            if color_range is not None:
                writer_output_dict['-color_range'] = color_range
                writer_input_dict['-color_range'] = color_range
            if color_space is not None:
                writer_output_dict['-colorspace'] = color_space
                writer_input_dict['-colorspace'] = color_space
            if color_primaries is not None:
                writer_output_dict['-color_primaries'] = color_primaries
                writer_input_dict['-color_primaries'] = color_primaries
            if color_transfer is not None:
                writer_output_dict['-color_trc'] = color_transfer
                writer_input_dict['-color_trc'] = color_transfer

            writer_output_dict['-sws_flags'] = 'full_chroma_int+bitexact+accurate_rnd'
            reader_output_dict['-sws_flags'] = 'full_chroma_int+bitexact+accurate_rnd'
            # writer_input_dict['-pix_fmt'] = 'bgr48le'
            # reader_output_dict = {'-pix_fmt': 'bgr48le'}

            # -s 1920x1080
            # writer_input_dict['-s'] = '1920x1080'
            # writer_output_dict['-s'] = '1920x1080'
            # writer_input_dict['-s'] = '1080x1920'
            # writer_output_dict['-s'] = '1080x1920'

            print(writer_input_dict)
            print(writer_output_dict)

            self.reader = skvideo.io.FFmpegReader(video_path, outputdict=reader_output_dict)
        else:
            
            # fps = 25
            codec_name = 'libx264'
            bit_rate=None
            pix_fmt = 'yuv420p'

            reader_output_dict = {'-r': str(fps)}
            writer_input_dict = {'-r': str(fps)}
            writer_output_dict = {'-pix_fmt': pix_fmt, '-r': str(fps), '-vcodec':str(codec_name)}
            # if bit_rate is not None:
            #     writer_output_dict['-b:v'] = bit_rate
            writer_output_dict['-crf'] = '17'

            # if video has alpha channel, convert to bgra, uint16 to process
            if pix_fmt.startswith('yuva'):
                writer_input_dict['-pix_fmt'] = 'bgra64le'
                reader_output_dict['-pix_fmt'] = 'bgra64le'
            elif pix_fmt.endswith('le'):
                writer_input_dict['-pix_fmt'] = 'bgr48le'
                reader_output_dict['-pix_fmt'] = 'bgr48le'
            else:
                writer_input_dict['-pix_fmt'] = 'bgr24'
                reader_output_dict['-pix_fmt'] = 'bgr24'

            writer_output_dict['-sws_flags'] = 'full_chroma_int+bitexact+accurate_rnd'
            print(writer_input_dict)
            print(writer_output_dict)

        if output_video_path is not None:
            self.writer = skvideo.io.FFmpegWriter(output_video_path, inputdict=writer_input_dict, outputdict=writer_output_dict, verbosity=1)

    def getframes(self):
        return self.reader.nextFrame()

    def writeframe(self, frame):
        if frame is None:
            self.writer.close()
        else:
            self.writer.writeFrame(frame)

def seed_everything(seed):
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)


def import_filename(filename):
    spec = importlib.util.spec_from_file_location("mymodule", filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def delete_additional_ckpt(base_path, num_keep):
    dirs = []
    for d in os.listdir(base_path):
        if d.startswith("checkpoint-"):
            dirs.append(d)
    num_tot = len(dirs)
    if num_tot <= num_keep:
        return
    # ensure ckpt is sorted and delete the ealier!
    del_dirs = sorted(dirs, key=lambda x: int(x.split("-")[-1]))[: num_tot - num_keep]
    for d in del_dirs:
        path_to_dir = osp.join(base_path, d)
        if osp.exists(path_to_dir):
            shutil.rmtree(path_to_dir)


def save_videos_from_pil(pil_images, path, fps=8):
    save_fmt = Path(path).suffix
    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = pil_images[0].size

    if save_fmt == ".mp4":
        video_cap = VideoUtils(output_video_path=path, fps=fps)
        for pil_image in pil_images:
            image_cv2 = np.array(pil_image)[:,:,[2,1,0]]
            video_cap.writeframe(image_cv2)
        video_cap.writeframe(None)

    elif save_fmt == ".gif":
        pil_images[0].save(
            fp=path,
            format="GIF",
            append_images=pil_images[1:],
            save_all=True,
            duration=(1 / fps * 1000),
            loop=0,
            optimize=False,
            lossless=True
        )
    else:
        raise ValueError("Unsupported file type. Use .mp4 or .gif.")


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    height, width = videos.shape[-2:]
    outputs = []

    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)  # (c h w)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)  # (h w c)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        x = Image.fromarray(x)

        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_videos_from_pil(outputs, path, fps)


def read_frames(video_path):
    container = av.open(video_path)

    video_stream = next(s for s in container.streams if s.type == "video")
    frames = []
    for packet in container.demux(video_stream):
        for frame in packet.decode():
            image = Image.frombytes(
                "RGB",
                (frame.width, frame.height),
                frame.to_rgb().to_ndarray(),
            )
            frames.append(image)

    return frames


def get_fps(video_path):
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == "video")
    fps = video_stream.average_rate
    container.close()
    return fps
