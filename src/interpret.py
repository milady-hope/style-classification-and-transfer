import torch
import torch.nn as nn

from .data import clean_text


class CNNEmbeddingWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, embeddings):
        self.model.lstm.train()
        x = embeddings.transpose(1, 2)
        outs = [conv(x) for conv in self.model.convs]
        ml = min(o.size(2) for o in outs)
        x = torch.cat([o[:, :, :ml] for o in outs], 1).transpose(1, 2)
        x, _ = self.model.lstm(x)
        w = torch.softmax(self.model.attn(x).squeeze(-1), 1)
        x = torch.bmm(w.unsqueeze(1), x).squeeze(1)
        return self.model.fc(self.model.drop(x))


def texts_to_tensor(texts, vocab, max_len: int = 1500, device=None):
    all_ids = []
    for t in texts:
        chars = list(clean_text(t).lower())
        ids = [vocab.get(c, 1) for c in chars][:max_len]
        ids += [0] * (max_len - len(ids))
        all_ids.append(ids)
    out = torch.tensor(all_ids, dtype=torch.long)
    return out.to(device) if device is not None else out
