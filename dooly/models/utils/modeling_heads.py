import torch
import torch.nn as nn


class SpanPredictionHead(nn.Module):
    """Head for span prediction tasks.
    Can be viewed as a 2-class output layer that is applied to every position.
    """

    def __init__(self, config):
        assert config.num_labels == 2
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.span_head_inner_dim)
        self.dropout = nn.Dropout(p=config.span_head_dropout)
        self.out_proj = nn.Linear(config.span_head_inner_dim, config.num_labels)

    def forward(self, features, **kwargs):
        x = features  # take features across ALL positions
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x  # B x T x C, but softmax should be taken over T


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class DependencyParseHead(nn.Module):
    """Head for sequence tagging tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.head_attn_pre = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.classifier_num_attention_heads,
            dropout=classifier_dropout,
        )
        self.head_attn_post = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=1,
            dropout=classifier_dropout,
        )

    def forward(self, features, masks=None, **kwargs):
        x = features
        x = self.dropout(x)

        x2 = x.permute(1, 0, 2)
        # https://github.com/kakaobrain/pororo/blob/master/pororo/models/brainbert/PoSLaBERTa.py#L110
        # deprecated! attention_mask vs masks
        # attention_mask = [[1,1,1,1],[1,1,1,0]]
        # masks = [[0,0,0,0],[0,0,0,1]]
        masks = (masks == 0).to(torch.bool)
        x3, _ = self.head_attn_pre(x2, x2, x2, key_padding_mask=masks)
        # x3 = [max_len, bsz, hidden_dim]
        _, attn = self.head_attn_post(x3, x3, x3, key_padding_mask=masks)
        # attn = [bsz, max_len, max_len]

        x = self.dense(x)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        label = self.out_proj(x)

        return attn, label
