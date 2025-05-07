import torch
import torch.nn as nn
import torch.nn.functional as F
from ...models.audio_adapter.audio_proj import AudioProjModel as FeatureProjModel

class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        self.emo_linear = FeatureProjModel(seq_len=1, blocks=1, channels=256, intermediate_dim=1024, output_dim=1024, context_tokens=32)
        self.kv_tokens_linear = FeatureProjModel(seq_len=1, blocks=1, channels=1024, intermediate_dim=1024, output_dim=1024, context_tokens=32)

        self.codebook = emotion_bank(src_dim=1024, codebook_size=64)
        
        self.attention = QKVAttention(input_dim=1024, output_dim=1024, num_heads=8)
        self.classifier = Classifier(input_dim=1280, num_classes=22)




    def forward(self, emo_prompts, emo_prompt_mask=None, retrieval=True):

        emo_retrieval, vq_loss, encoding_indices = self.codebook(emo_prompts)

        if emo_prompt_mask is not None:
            emo_prompts = torch.where(
                emo_prompt_mask, torch.zeros_like(emo_prompts), emo_prompts)

            emo_retrieval = torch.where(
                emo_prompt_mask, torch.zeros_like(emo_retrieval), emo_retrieval)
         
        emo_prompts_q = self.emo_linear(emo_prompts)

        if retrieval:
            kv_tokens = self.kv_tokens_linear(emo_retrieval)
        else:
            f = emo_prompts.shape[1]
            num, d = self.codebook.codebook.weight.data.shape
            emo_retrieval = self.codebook.codebook.weight.data.view(1,num,1,1,d)
            kv_tokens = self.kv_tokens_linear(emo_retrieval)
            kv_tokens = kv_tokens.view(1,1,-1,1024).expand(1, f, -1, 1024)
            vq_loss = torch.zeros_like(vq_loss)
        
        final_emo_prompts , attn_weights = self.attention(emo_prompts_q, kv_tokens)

        return final_emo_prompts, vq_loss

class QKVAttention(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8):
        super(QKVAttention, self).__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        
        self.fc_q = nn.Linear(input_dim, output_dim)
        self.fc_k = nn.Linear(input_dim, output_dim)
        self.fc_v = nn.Linear(input_dim, output_dim)
        


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


        attn_output = attn_output.transpose(2, 3).contiguous().view(attn_output.size(0), attn_output.size(1), attn_output.size(3), -1)



        return attn_output, attn_weights
    
class emotion_bank(nn.Module):
    def __init__(self, src_dim=1024, codebook_size=512):
        super(emotion_bank, self).__init__()
        self.src_dim = src_dim
        self._commitment_cost = 0.25

        self.fc = nn.Linear(256, src_dim)
        self.codebook = nn.Embedding(codebook_size, src_dim)
        self.codebook_size = codebook_size
        self.codebook.weight.data.uniform_(-1/self.codebook_size, 1/self.codebook_size)

 

    def quantize(self, z):
        
        z = self.fc(z)
        b, l, d1, d2, c = z.shape
        flat_z = z.reshape(-1, c)
        # Calculate distances
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True)
                    + torch.sum(self.codebook.weight**2, dim=1)
                    - 2 * torch.matmul(flat_z, self.codebook.weight.t()))


        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self.codebook_size, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.codebook.weight).view(b, l, d1, d2, c)
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = z + (quantized - z).detach()



        return quantized, loss, encoding_indices

    def forward(self, z):
        
        z, loss, encoding_indices= self.quantize(z)

        return z, loss, encoding_indices
    

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = x.view(x.shape[0],-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


    
    