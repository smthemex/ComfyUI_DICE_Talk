# ComfyUI_DICE_Talk
Use [DICE-Talk](https://github.com/toto222/DICE-Talk) in ComfyUI，which is a method about 'Correlation-Aware Emotional Talking Portrait Generation'.

# Will Update codes tomorrow,coming soon.


# 1. Installation

In the ./ComfyUI/custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_DICE_Talk.git
```
---

# 2. Requirements  

```
pip install -r requirements.txt
```

# 3.Model
* 3.1.1 download  checkpoints  from [EEEELY/DICE-Talk](https://huggingface.co/EEEELY/DICE-Talk/tree/main) 从抱脸下载必须的模型,文件结构如下图
* 3.1.2 download [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny/tree/main)
```
--  ComfyUI/models/dice_talk/
    |-- audio_linear.pth
    |-- emo_model.pth
    |-- pose_guider.pth
    |-- unet.pth
    |-- yoloface_v5m.pt  #can use sonic 可以用sonic的，不需要复制
    |-- whisper-tiny/  #can use sonic 可以用sonic的，不需要复制
        |--config.json
        |--model.safetensors
        |--preprocessor_config.json
    |-- RIFE/  #can use sonic 可以用sonic的，不需要复制
        |--flownet.pkl
```
*  3.2 SVD checkpoints  [svd_xt.safetensors](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)  or [svd_xt_1_1.safetensors](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1)    

```
--   ComfyUI/models/checkpoints
    ├── svd_xt.safetensors  or  svd_xt_1_1.safetensors
```

# 4.Example

![](https://github.com/smthemex/ComfyUI_DICE_Talk/blob/main/example_workflows/example.png)


# 5 .Citation
```
@misc{tan2025disentangleidentitycooperateemotion,
      title={Disentangle Identity, Cooperate Emotion: Correlation-Aware Emotional Talking Portrait Generation}, 
      author={Weipeng Tan and Chuming Lin and Chengming Xu and FeiFan Xu and Xiaobin Hu and Xiaozhong Ji and Junwei Zhu and Chengjie Wang and Yanwei Fu},
      year={2025},
      eprint={2504.18087},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.18087}, 
}
```
