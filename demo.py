import os
import argparse
from dice_talk import DICE_Talk
pipe = DICE_Talk(0)


parser = argparse.ArgumentParser()
parser.add_argument('--image_path')
parser.add_argument('--audio_path')
parser.add_argument('--emotion_path')
parser.add_argument('--output_path')
parser.add_argument('--ref_scale', type=float, default=3.0)
parser.add_argument('--emo_scale', type=float, default=6.0)
parser.add_argument('--crop', action='store_true')
parser.add_argument('--seed', type=int, default=None)

args = parser.parse_args()


face_info = pipe.preprocess(args.image_path, expand_ratio=0.5)
print(face_info)
if face_info['face_num'] >= 0:
    if args.crop:
        crop_image_path = args.image_path + '.crop.png'
        pipe.crop_image(args.image_path, crop_image_path, face_info['crop_bbox'])
        args.image_path = crop_image_path
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    pipe.process(args.image_path, args.audio_path, args.emotion_path, args.output_path, min_resolution=512, inference_steps=25, ref_scale=args.ref_scale, emo_scale=args.emo_scale, seed=args.seed)