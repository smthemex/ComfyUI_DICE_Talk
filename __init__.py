"""ComfyUI_DICE_Talk_Native: DICE-Talk (emotional audio-driven portrait) on the
comfy native stack.

Native SVD checkpoint VAE / CLIP_VISION / sampling semantics + native VAEDecode.
Self-contained: DiceTalkUNet (2 IP-Adapter branches, loaded 1:1 from dice_unet.pth
with strict=True), PoseGuider, whisper-mini encoder, audio/emotion projections,
sliding-window 3-branch CFG sampler. No diffusers / transformers.
"""
import os
import sys

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

plugin_path = os.path.dirname(os.path.realpath(__file__))
if plugin_path not in sys.path:
    sys.path.insert(0, plugin_path)

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: F401
except Exception as e:
    print("[ComfyUI_DICE_Talk_Native] import failed:", e)
    import traceback

    traceback.print_exc()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
