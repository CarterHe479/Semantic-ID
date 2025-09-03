import torch, torch.nn as nn

class SemanticIDDecoder(nn.Module):
    def __init__(self, vocab_sizes, d_model=512, n_layers=6, n_heads=8):
        super().__init__()
        # RQ 系：每个位一个小词表 → 多头输出；或拼成大的“组合词表”
        self.embs = nn.ModuleList([nn.Embedding(v, d_model) for v in vocab_sizes])
        self.tok_type = nn.Embedding(8, d_model)  # 行为类型等
        encoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, batch_first=True)
        self.dec = nn.TransformerDecoder(encoder_layer, num_layers=n_layers)
        self.heads = nn.ModuleList([nn.Linear(d_model, v) for v in vocab_sizes])

    def forward(self, hist_tokens, token_types):
        H = sum([e(hist_tokens[:,:,i]) for i,e in enumerate(self.embs)]) + self.tok_type(token_types)
        Y = self.dec(H, H)  # 自回归时配 causal mask
        logits = [head(Y[:, -1, :]) for head in self.heads]  # 预测下一 item 的各码位
        return logits
