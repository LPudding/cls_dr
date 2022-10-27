import torch
import torch.nn as nn
import random

__all__ = ['DHL']


# def __init__(self, importance_rate=0.70, drop_or_highlight_rate=0.85, drop_threshold=0.8, highlight_threshold=0.5):
class DHL(nn.Module):
    def __init__(self, importance_rate=0.60, drop_or_highlight_rate=0.85, drop_threshold=0.8, highlight_threshold=0.5):
        super(DHL, self).__init__()
        if not (0 <= importance_rate <= 1):
            raise ValueError("importance_rate must be in range [0, 1].")
        if not (0 <= drop_or_highlight_rate <= 1):
            raise ValueError("drop_or_highlight_rate must be in range [0, 1].")
        if not (0 <= drop_threshold <= 1):
            raise ValueError("drop_threshold must be in range [0, 1].")
        if not (0 <= highlight_threshold <= 1):
            raise ValueError("highlight_threshold must be in range [0, 1].")
        self.importance_rate = importance_rate
        self.drop_or_highlight_rate = drop_or_highlight_rate
        self.drop_threshold = drop_threshold
        self.highlight_threshold = highlight_threshold
        self.attention = None
        self.drop_mask = None

    def forward(self, input_):
        if not self.training:
            return input_
        else:
            attention = torch.mean(input_, dim=1, keepdim=True)
            importance_map = torch.sigmoid(attention)
            drop_mask = self._drop_mask(attention)
            highlight_mask = self._highlight_mask(attention)
            selected_map = self._select_map(importance_map, drop_mask, highlight_mask)
            return input_.mul(selected_map)

    def _select_map(self, importance_map, drop_mask, highlight_mask):
        p = random.uniform(0, 1)
        if p <= self.importance_rate: return importance_map
        elif p > self.importance_rate and p < self.drop_or_highlight_rate: return drop_mask
        elif p >= self.drop_or_highlight_rate: return highlight_mask

    def _drop_mask(self, attention):
        b_size = attention.size(0)
        max_val, _ = torch.max(attention.view(b_size, -1), dim=1, keepdim=True)
        thr_val = max_val * self.drop_threshold
        thr_val = thr_val.view(b_size, 1, 1, 1)
        return (attention < thr_val).float()

    def _highlight_mask(self, attention):
        b_size = attention.size(0)
        max_val, _ = torch.max(attention.view(b_size, -1), dim=1, keepdim=True)
        thr_val = max_val * self.highlight_threshold
        thr_val = thr_val.view(b_size, 1, 1, 1)
        return (attention > thr_val).float()

