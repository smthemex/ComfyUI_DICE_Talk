"""Audio / emotion front-ends for DICE-Talk (torch only).

- AudioProjModel: identical to Sonic's audio2token projection (seq_len 10 x 5 x 384
  -> 32 x 1024) but DICE uses its own `audio_linear.pth`.
- EmotionModel: emotion prior (256) -> codebook lookup / kv tokens -> 32 x 1024
  context tokens, matching `src/models/emotion_adapter/emo.py`.
"""
import torch
from torch import nn
import torch.nn.functional as F


class AudioProjModel(nn.Module):
    def __init__(self, seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=1024,
                 context_tokens=32, operations=None):
        super().__init__()
        if operations is None:
            import torch.nn as operations
        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        self.proj1 = operations.Linear(self.input_dim, intermediate_dim)
        self.proj2 = operations.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = operations.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = operations.LayerNorm(output_dim)

    def forward(self, audio_embeds):
        # [bz, f, w, b, c] -> [(bz f), w, b, c]
        video_length = audio_embeds.shape[1]
        audio_embeds = audio_embeds.reshape(-1, *audio_embeds.shape[2:])
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.reshape(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(batch_size, self.context_tokens, self.output_dim)
        context_tokens = self.norm(context_tokens)
        return context_tokens.reshape(-1, video_length, *context_tokens.shape[1:])


class EmotionBank(nn.Module):
    """VQ codebook (`emotion_bank` in the official code; inference only)."""

    def __init__(self, src_dim=1024, codebook_size=64, operations=None):
        super().__init__()
        if operations is None:
            import torch.nn as operations
        self.src_dim = src_dim
        self._commitment_cost = 0.25
        self.fc = operations.Linear(256, src_dim)
        self.codebook = nn.Embedding(codebook_size, src_dim)
        self.codebook_size = codebook_size

    def quantize(self, z):
        z = self.fc(z)
        b, l, d1, d2, c = z.shape
        flat_z = z.reshape(-1, c)
        distances = (torch.sum(flat_z ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebook.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_z, self.codebook.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.codebook.weight).view(b, l, d1, d2, c)
        # straight-through, exactly as the original
        quantized = z + (quantized - z).detach()
        return quantized, encoding_indices

    def forward(self, z):
        quantized, idx = self.quantize(z)
        return quantized, torch.zeros((), device=z.device), idx


class QKVAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, operations=None):
        super().__init__()
        if operations is None:
            import torch.nn as operations
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.fc_q = operations.Linear(input_dim, output_dim)
        self.fc_k = operations.Linear(input_dim, output_dim)
        self.fc_v = operations.Linear(input_dim, output_dim)

    def forward(self, x, y):
        Q = self.fc_q(x)
        K = self.fc_k(y)
        V = self.fc_v(y)

        Q = Q.view(Q.size(0), Q.size(1), Q.size(2), self.num_heads, -1).transpose(2, 3)
        K = K.view(K.size(0), K.size(1), K.size(2), self.num_heads, -1).transpose(2, 3)
        V = V.view(V.size(0), V.size(1), V.size(2), self.num_heads, -1).transpose(2, 3)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(2, 3).contiguous().view(
            attn_output.size(0), attn_output.size(1), attn_output.size(3), -1)
        return attn_output, attn_weights


class EmotionModel(nn.Module):
    """dice emo_model.pth; the `classifier` head is training-only but its keys exist,
    so it is instantiated to keep load_state_dict(strict=True) honest."""

    def __init__(self, operations=None):
        super().__init__()
        if operations is None:
            import torch.nn as operations
        self.emo_linear = AudioProjModel(seq_len=1, blocks=1, channels=256, intermediate_dim=1024,
                                         output_dim=1024, context_tokens=32, operations=operations)
        self.kv_tokens_linear = AudioProjModel(seq_len=1, blocks=1, channels=1024, intermediate_dim=1024,
                                               output_dim=1024, context_tokens=32, operations=operations)
        self.codebook = EmotionBank(src_dim=1024, codebook_size=64, operations=operations)
        self.attention = QKVAttention(input_dim=1024, output_dim=1024, num_heads=8, operations=operations)
        self.classifier = Classifier(input_dim=1280, num_classes=22, operations=operations)

    def forward(self, emo_prompts, emo_prompt_mask=None, retrieval=False):
        emo_retrieval, _, _ = self.codebook(emo_prompts)

        if emo_prompt_mask is not None:
            emo_prompts = torch.where(emo_prompt_mask, torch.zeros_like(emo_prompts), emo_prompts)
            emo_retrieval = torch.where(emo_prompt_mask, torch.zeros_like(emo_retrieval), emo_retrieval)

        emo_prompts_q = self.emo_linear(emo_prompts)

        if retrieval:
            kv_tokens = self.kv_tokens_linear(emo_retrieval)
        else:
            f = emo_prompts.shape[1]
            num, d = self.codebook.codebook.weight.data.shape
            emo_retrieval = self.codebook.codebook.weight.data.view(1, num, 1, 1, d)
            kv_tokens = self.kv_tokens_linear(emo_retrieval)
            kv_tokens = kv_tokens.view(1, 1, -1, 1024).expand(1, f, -1, 1024)

        final_emo_prompts, _ = self.attention(emo_prompts_q, kv_tokens)
        return final_emo_prompts, torch.zeros((), device=emo_prompts.device)


class Classifier(nn.Module):
    """Unused at inference; present so the checkpoint loads strictly."""

    def __init__(self, input_dim, num_classes, operations=None):
        super().__init__()
        if operations is None:
            import torch.nn as operations
        self.fc1 = operations.Linear(input_dim, 512)
        self.fc2 = operations.Linear(512, 256)
        self.fc3 = operations.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)
