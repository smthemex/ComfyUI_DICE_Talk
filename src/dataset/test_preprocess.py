import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import CLIPImageProcessor
import librosa


def process_bbox(bbox, expand_radio, height, width):
    """
    raw_vid_path:
    bbox: format: x1, y1, x2, y2
    radio: expand radio against bbox size
    height,width: source image height and width
    """

    def expand(bbox, ratio, height, width):
        
        bbox_h = bbox[3] - bbox[1]
        bbox_w = bbox[2] - bbox[0]
        
        expand_x1 = max(bbox[0] - ratio * bbox_w, 0)
        expand_y1 = max(bbox[1] - ratio * bbox_h, 0)
        expand_x2 = min(bbox[2] + ratio * bbox_w, width)
        expand_y2 = min(bbox[3] + ratio * bbox_h, height)

        return [expand_x1,expand_y1,expand_x2,expand_y2]

    def to_square(bbox_src, bbox_expend, height, width):

        h = bbox_expend[3] - bbox_expend[1]
        w = bbox_expend[2] - bbox_expend[0]
        c_h = (bbox_expend[1] + bbox_expend[3]) / 2
        c_w = (bbox_expend[0] + bbox_expend[2]) / 2

        c = min(h, w) / 2

        c_src_h = (bbox_src[1] + bbox_src[3]) / 2
        c_src_w = (bbox_src[0] + bbox_src[2]) / 2

        s_h, s_w = 0, 0
        if w < h:
            d = abs((h - w) / 2)
            s_h = min(d, abs(c_src_h-c_h))
            s_h = s_h if  c_src_h > c_h else s_h * (-1)
        else:
            d = abs((h - w) / 2)
            s_w = min(d, abs(c_src_w-c_w))
            s_w = s_w if  c_src_w > c_w else s_w * (-1)


        c_h = (bbox_expend[1] + bbox_expend[3]) / 2 + s_h
        c_w = (bbox_expend[0] + bbox_expend[2]) / 2 + s_w

        square_x1 = c_w - c
        square_y1 = c_h - c
        square_x2 = c_w + c
        square_y2 = c_h + c

        x1, y1, x2, y2 = square_x1, square_y1, square_x2, square_y2
        ww = x2 - x1
        hh = y2 - y1
        cc_x = (x1 + x2)/2
        cc_y = (y1 + y2)/2
        # 1:1
        ww = hh = min(ww, hh)
        x1, x2 = round(cc_x - ww/2), round(cc_x + ww/2)
        y1, y2 = round(cc_y - hh/2), round(cc_y + hh/2) 

        return [round(x1), round(y1), round(x2), round(y2)]


    bbox_expend = expand(bbox, expand_radio, height=height, width=width)
    processed_bbox = to_square(bbox, bbox_expend, height=height, width=width)

    return processed_bbox


def get_audio_feature(audio_path, feature_extractor,infer_duration):
    audio_input, sampling_rate = librosa.load(audio_path, sr=16000,duration=infer_duration)
    assert sampling_rate == 16000

    audio_features = []
    window = 750*640
    for i in range(0, len(audio_input), window):
        audio_feature = feature_extractor(audio_input[i:i+window], 
                                        sampling_rate=sampling_rate, 
                                        return_tensors="pt", 
                                        ).input_features
        audio_features.append(audio_feature)
    audio_features = torch.cat(audio_features, dim=-1)
    return audio_features, len(audio_input) // 640

def image_audio_emo_to_tensor(align_instance, feature_extractor, imSrc_, audio_path, emotion_path,infer_duration, limit=100, image_size=512, area=1.25):
    
    clip_processor = CLIPImageProcessor()
    
    to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    mask_to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])
    

    #imSrc_ = Image.open(image_path).convert('RGB')
    w, h = imSrc_.size
    
    _, _, bboxes_list = align_instance(np.array(imSrc_)[:,:,[2,1,0]], maxface=True)

    if len(bboxes_list) == 0:
        return None
    bboxSrc = bboxes_list[0]

    x1, y1, ww, hh = bboxSrc
    x2, y2 = x1 + ww, y1 + hh

    mask_img = np.zeros_like(np.array(imSrc_))
    ww, hh = (x2-x1) * area, (y2-y1) * area
    center = [(x2+x1)//2, (y2+y1)//2]
    x1 = max(center[0] - ww//2, 0)
    y1 = max(center[1] - hh//2, 0)
    x2 = min(center[0] + ww//2, w)
    y2 = min(center[1] + hh//2, h)
    mask_img[int(y1):int(y2), int(x1):int(x2)] = 255
    mask_img = Image.fromarray(mask_img)
    
    w, h = imSrc_.size
    scale = image_size / min(w, h)
    new_w = round(w * scale / 64) * 64
    new_h = round(h * scale / 64) * 64
    if new_h != h or new_w != w:
        imSrc = imSrc_.resize((new_w, new_h), Image.LANCZOS)
        mask_img = mask_img.resize((new_w, new_h), Image.LANCZOS)
    else:
        imSrc = imSrc_

    # clip_image = clip_processor(
    #         images=imSrc.resize((224, 224), Image.LANCZOS), return_tensors="pt"
    #     ).pixel_values[0]
    audio_input, audio_len = get_audio_feature(audio_path, feature_extractor,infer_duration)

    audio_len = min(limit, audio_len)

    emotion_prior = torch.from_numpy(np.load(emotion_path, allow_pickle=True).item()['mu']).unsqueeze(0).unsqueeze(0)

    sample = dict(
                face_mask=mask_to_tensor(mask_img),
                ref_img=to_tensor(imSrc),
                #clip_images=clip_image,
                audio_feature=audio_input[0],
                audio_len=audio_len,
                emo_feature=emotion_prior,
            )

    return sample