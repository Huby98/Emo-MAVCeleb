import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except Exception:
    AutoModel = None  # only needed at runtime; smoke tests may stub this out


class SSLEmotionStudent(nn.Module):
    """
    Self-supervised audio student for cross-modal emotion distillation.

    Drop-in replacement for EmoVGGVoxStudent. Same interface:
      input : raw waveform (B, S)      -- 16 kHz mono, S = sr*dur_s (e.g. 64000)
      output: logits       (B, num_classes)

    A pretrained speech encoder (WavLM / wav2vec2 / XLS-R) produces per-frame
    contextual features; we take a learnable weighted sum over its hidden
    layers (SUPERB-style -- emotion peaks in mid/upper layers), mean-pool over
    time, and map to the emotion classes with a small MLP head.

    The encoder is FROZEN by default: only `layer_weights` and `head` train,
    so the comparison against VGGVox isolates the audio representation.
    """

    def __init__(self, ckpt="microsoft/wavlm-base-plus", num_classes=8,
                 freeze_encoder=True):
        super().__init__()
        if AutoModel is None:
            raise ImportError("transformers is required: pip install transformers")
        self.encoder = AutoModel.from_pretrained(ckpt, output_hidden_states=True, use_safetensors=True)
        hidden = self.encoder.config.hidden_size
        n_layers = self.encoder.config.num_hidden_layers + 1   # +1 for embeddings
        self.layer_weights = nn.Parameter(torch.ones(n_layers))
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
        self.head = nn.Sequential(
            nn.Linear(hidden, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def train(self, mode=True):
        # Keep a frozen encoder in eval() so its dropout/LN stay deterministic,
        # while the head trains normally.
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def _encode(self, wav, attention_mask):
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(wav, attention_mask=attention_mask)
        return self.encoder(wav, attention_mask=attention_mask)

    def forward(self, wav, attention_mask=None):
        out = self._encode(wav, attention_mask)
        hs = torch.stack(out.hidden_states, dim=0)          # (L, B, T, H)
        w = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
        x = (hs * w).sum(dim=0)                              # (B, T, H)
        if attention_mask is not None:
            m = attention_mask.unsqueeze(-1).float()
            x = (x * m).sum(1) / m.sum(1).clamp(min=1)      # masked mean pool
        else:
            x = x.mean(dim=1)                               # mean pool over time
        return self.head(x)                                 # (B, num_classes)
