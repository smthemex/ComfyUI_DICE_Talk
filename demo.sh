#!/bin/bash

image_path=examples/img/female.png
audio_path=examples/wav/female.wav
emotion_path=examples/emo/happy.npy
output_path=results/output.mp4

python3 demo.py --image_path $image_path --audio_path $audio_path --emotion_path $emotion_path --output_path $output_path