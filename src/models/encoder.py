import torch, torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2", d=256):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        self.proj = nn.Linear(self.backbone.config.hidden_size, d)
    def forward(self, input_ids, attn_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state[:,0]
        return self.proj(out)

class FuseEncoder(nn.Module):
    def __init__(self, d=256, img_d=384, txt_d=256, attr_d=64):
        super().__init__()
        self.txt = TextEncoder(d=txt_d)
        self.attr = nn.Sequential(nn.Linear(attr_d, d), nn.ReLU(), nn.Linear(d, d))
        self.proj_img = nn.Linear(img_d, d)
        self.out = nn.Linear(d*3, d)
    def forward(self, txt_h, img_h, attr):
        z = torch.cat([txt_h, self.proj_img(img_h), self.attr(attr)], dim=-1)
        return self.out(z)  # [B, d]
